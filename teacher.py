"""Frozen teacher embeddings, computed once: data-parallel inference, rank r embeds shard r on its own GPU.
Output per set: emb_dir/<set>.npy (fp16, unit norm, [N, dim]) + <set>_ids.json, in data.json order."""
import os
import time
import traceback

import numpy as np
import torch

from data import load_json, resolve_cfg, save_json
from ddp import barrier, cleanup, is_main, setup
from model import DTYPES, encode, gpu_mem, load_encoder

QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

TEACHER_DEFAULTS = dict(
    model="Qwen/Qwen3-Embedding-4B",   # hub id or local dir
    dim=256,                           # MRL truncation, then L2-normalise
    dtype="fp16",
    query_prompt=QUERY_PROMPT,         # queries only; passages get no prefix
    doc_max_len=128,
    query_max_len=64,                  # prompt (~20 tokens) + query
    batch_size=64,
    data_path="cache/data.json",
    emb_dir="cache/emb/teacher",
    sets=("corpus", "align", "val", "test"),
)


def shard_path(emb_dir, name, rank, world_size):
    return os.path.join(emb_dir, f"{name}.shard{rank}of{world_size}.npy")


def _set_args(cfg, name):
    """(prefix, max_len) for a set: the corpus is passages, every other set is queries."""
    return ("", cfg["doc_max_len"]) if name == "corpus" else (cfg["query_prompt"], cfg["query_max_len"])


def embed_main(rank, world_size, cfg):
    """One worker of ddp.launch(embed_main, cfg). Shards that already exist are skipped (resumable)."""
    cfg = resolve_cfg(TEACHER_DEFAULTS, cfg)
    device = setup(rank, world_size)
    print(f"[rank{rank}] joined process group on {device}{gpu_mem(device)}", flush=True)
    try:
        data = load_json(cfg["data_path"])
        os.makedirs(cfg["emb_dir"], exist_ok=True)
        todo = [s for s in cfg["sets"]
                if not all(os.path.exists(shard_path(cfg["emb_dir"], s, r, world_size)) for r in range(world_size))
                and not os.path.exists(os.path.join(cfg["emb_dir"], f"{s}.npy"))]
        print(f"[rank{rank}] sets to embed: {todo or 'none'}", flush=True)
        barrier()  # everyone decides `todo` before anyone writes
        if todo:
            if not is_main():
                print(f"[rank{rank}] waiting for rank 0 to load the model", flush=True)
                barrier()  # rank 0 downloads first, the others read the HF cache
            print(f"[rank{rank}] loading {cfg['model']} ({cfg['dtype']})", flush=True)
            model, tokenizer = load_encoder(cfg["model"], device, DTYPES[cfg["dtype"]])
            print(f"[rank{rank}] model loaded on {device}{gpu_mem(device)}", flush=True)
            if is_main():
                barrier()
            for name in todo:
                path = shard_path(cfg["emb_dir"], name, rank, world_size)
                if os.path.exists(path):
                    print(f"[rank{rank}] {name}: shard exists, skipped", flush=True)
                    continue
                texts = data[name]["texts"]
                lo, hi = _shard_bounds(len(texts), rank, world_size)
                prefix, max_len = _set_args(cfg, name)
                print(f"[rank{rank}] {name}: embedding texts {lo}:{hi} of {len(texts)}", flush=True)
                emb = encode(model, tokenizer, texts[lo:hi], device, cfg["dim"], max_len, prefix,
                             cfg["batch_size"], progress=True)
                np.save(path + ".tmp.npy", emb.numpy())
                os.replace(path + ".tmp.npy", path)
                print(f"[rank{rank}] {name}: saved {path}", flush=True)
            del model
        print(f"[rank{rank}] waiting for all ranks before merging", flush=True)
        barrier()
        if is_main():
            for name in cfg["sets"]:
                merge_shards(cfg["emb_dir"], name, world_size, data[name]["ids"])
        barrier()
        print(f"[rank{rank}] done", flush=True)
    except BaseException:
        traceback.print_exc()  # print now: cleanup() can hang while another rank waits, hiding this
        raise
    finally:
        cleanup()


def _shard_bounds(n, rank, world_size):
    per = -(-n // world_size)
    return min(rank * per, n), min((rank + 1) * per, n)


def merge_shards(emb_dir, name, world_size, ids):
    """Concatenate shards in rank order into <name>.npy + <name>_ids.json. No-op if already merged."""
    out = os.path.join(emb_dir, f"{name}.npy")
    if os.path.exists(out):
        return
    emb = np.concatenate([np.load(shard_path(emb_dir, name, r, world_size)) for r in range(world_size)])
    assert len(emb) == len(ids), f"{name}: {len(emb)} vectors for {len(ids)} ids"
    np.save(out + ".tmp.npy", emb)
    os.replace(out + ".tmp.npy", out)
    save_json(ids, os.path.join(emb_dir, f"{name}_ids.json"))
    print(f"[teacher] merged {name}: {emb.shape}")


def fp16_sanity(model_name, texts, dim, max_len, device, device_map_fp32=None):
    """fp16 vs fp32 embeddings of the same texts: mean/min cosine and NaN count. Loads one copy at a time.
    The 4B model in fp32 (16 GB) does not fit on one T4: pass device_map_fp32="auto" to spread it over both GPUs."""
    out = {}
    for dtype in ("fp32", "fp16"):
        model, tokenizer = load_encoder(model_name, device, DTYPES[dtype],
                                        device_map=device_map_fp32 if dtype == "fp32" else None)
        dev = next(model.parameters()).device
        out[dtype] = encode(model, tokenizer, texts, dev, dim, max_len, amp=False).float()
        del model
        free_gpu_memory()
    cos = (out["fp32"] * out["fp16"]).sum(-1)
    result = {"mean_cos": cos.mean().item(), "min_cos": cos.min().item(),
              "nan_rows": int((~torch.isfinite(out["fp16"])).any(-1).sum())}
    print(f"[fp16_sanity] {result}")
    return result


def time_embedding(cfg, data, device, world_size, n_sample=2000, budget_h=5.0):
    """Measure teacher tokens/s on a passage sample and project the full embedding run (all sets, world_size GPUs).
    Only reports; over budget means you shrink DATA yourself."""
    cfg = resolve_cfg(TEACHER_DEFAULTS, cfg)
    model, tokenizer = load_encoder(cfg["model"], device, DTYPES[cfg["dtype"]])
    sample = data["corpus"]["texts"][:n_sample]
    encode(model, tokenizer, sample[:cfg["batch_size"]], device, cfg["dim"], cfg["doc_max_len"])  # warm-up
    _sync(device)
    t0 = time.perf_counter()
    encode(model, tokenizer, sample, device, cfg["dim"], cfg["doc_max_len"], batch_size=cfg["batch_size"])
    _sync(device)
    tokens_per_s = _count_tokens(tokenizer, sample, "", cfg["doc_max_len"]) / (time.perf_counter() - t0)
    total = sum(_count_tokens(tokenizer, data[s]["texts"], *_set_args(cfg, s)) for s in cfg["sets"])
    projected_h = total / tokens_per_s / world_size / 3600
    del model
    free_gpu_memory()
    result = {"tokens_per_s": tokens_per_s, "total_tokens": total, "projected_h": projected_h, "budget_h": budget_h}
    print(f"[time_embedding] {tokens_per_s:,.0f} tok/s per GPU, {total:,} tokens -> {projected_h:.2f} h "
          f"on {world_size} GPU(s) (budget {budget_h} h{', OVER' if projected_h > budget_h else ''})")
    return result


def _count_tokens(tokenizer, texts, prefix, max_len):
    ids = tokenizer([prefix + t for t in texts], truncation=True, max_length=max_len)["input_ids"]
    return sum(len(x) for x in ids)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def free_gpu_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

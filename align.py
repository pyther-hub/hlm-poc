"""Alignment fine-tuning: a frozen student (Qwen3-Embedding-0.6B) + adapter (LoRA on every decoder linear, linear
head to the teacher dim) trained so its query vectors match the teacher's cached query vectors (emb/teacher/align.npy).
DDP: one process per GPU, each takes its slice of every global batch; only the adapter has gradients.
Checkpoints are adapter-only (model.save_adapter): <out_dir>/step_<n>/ (+ trainer_state.pt, resumable) and
<out_dir>/final/, which load_encoder / evaluate_model accept like a model dir.
The training loop (run) and its helpers are shared with refine.py; each phase only supplies its batch loss."""
import contextlib
import functools
import glob
import json
import math
import os
import shutil
import time
import traceback

import numpy as np
import torch

from data import load_emb, load_json, resolve_cfg
from ddp import barrier, cleanup, get_rank, is_main, reduce_mean, setup, wrap_ddp
from evaluation import score
from model import (TRAINER_STATE, attach_adapter, attach_head, embed, encode, gpu_mem, load_adapter_weights, load_encoder,
                   save_adapter, tokenize, trainable_params)

ALIGN_DEFAULTS = dict(
    model="Qwen/Qwen3-Embedding-0.6B",  # student base: hub id or local dir; frozen, fp32 (amp does the speed-up)
    dim=256,                            # head output = teacher index dim
    full=False,                         # True: no LoRA, train every base weight + head (lora_* ignored)
    lora_rank=16,
    lora_alpha=32,                      # scale = alpha / rank
    lora_dropout=0.05,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    query_prefix="",                    # student query prompt
    max_query_len=32,
    loss="l2",                          # name in LOSSES
    loss_kwargs={},
    lr=2e-4,                            # LoRA params (every base weight if full)
    head_lr=1e-3,                       # head params
    weight_decay=0.0,
    warmup_ratio=0.05,                  # linear warmup, then cosine decay to 0
    epochs=1,
    batch_size=128,                     # per GPU; global batch = batch_size * world_size * grad_accum
    grad_accum=1,                       # micro-batches per optimizer step
    grad_clip=1.0,
    amp=True,                           # fp16 autocast + GradScaler on CUDA (T4 has no bf16)
    seed=0,
    log_every=20,                       # steps; train loss averaged over ranks
    eval_every=200,                     # steps; val loss + retrieval on rank 0, 0 = only at the end
    patience=0,                         # evals without a val_loss drop before stopping, best weights -> final; 0 = off
    save_every=500,                     # steps; resumable adapter checkpoint, 0 = none
    keep_last=2,                        # step checkpoints kept
    data_path="cache/data.json",
    teacher_emb_dir="cache/emb/teacher",
    out_dir="runs/align",
)


def l2_loss(student, teacher):
    """Squared L2 distance between unit vectors, mean over the batch (= 2 - 2 cos)."""
    return (student - teacher).pow(2).sum(-1).mean()


LOSSES = {"l2": l2_loss}


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def epoch_order(n, seed, epoch):
    """Same permutation on every rank, and again on resume."""
    return np.random.default_rng([seed, epoch]).permutation(n)


def step_checkpoints(out_dir):
    """Finished step_<n> dirs, oldest first (zero-padded names sort by step)."""
    return sorted(p for p in glob.glob(os.path.join(out_dir, "step_*")) if not p.endswith(".tmp"))


def align_main(rank, world_size, cfg):
    """One worker of ddp.launch(align_main, cfg). Resumes from the latest step checkpoint; no-op if final/ exists."""
    cfg = resolve_cfg(ALIGN_DEFAULTS, cfg)
    device = setup(rank, world_size)
    print(f"[rank{rank}] joined process group on {device}{gpu_mem(device)}", flush=True)
    try:
        if not final_exists(cfg["out_dir"]):
            train(rank, world_size, cfg, device)
    except BaseException:
        traceback.print_exc()  # print now: cleanup() can hang while another rank waits, hiding this
        raise
    finally:
        cleanup()


def train(rank, world_size, cfg, device):
    data = load_json(cfg["data_path"])
    texts = data["align"]["texts"]
    #targets [N, d] teacher vectors for the align queries.
    #ids [N] query ids for the align queries, must match data["align"]["ids"].
    targets, ids = load_emb(cfg["teacher_emb_dir"], "align")
    assert ids == data["align"]["ids"], "align: teacher vectors don't match data.json"
    assert targets.shape[1] == cfg["dim"], f"teacher dim {targets.shape[1]} != cfg dim {cfg['dim']}"

    #convert the targets to a torch tensor of type float32 for training
    targets = torch.from_numpy(targets.astype(np.float32))

    
    model, tokenizer = load_rank0_first(cfg["model"], device)
    torch.manual_seed(cfg["seed"])  # same adapter init on every rank (DDP also broadcasts it from rank 0)
    if cfg["full"]:
        attach_head(model, cfg["dim"])
        model.requires_grad_(True)
        adapter_cfg = dict(base=cfg["model"], out_dim=cfg["dim"], full=True)
    else:
        attach_adapter(model, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["lora_targets"],
                       cfg["dim"])
        adapter_cfg = dict(base=cfg["model"], rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
                           dropout=cfg["lora_dropout"], targets=list(cfg["lora_targets"]), out_dim=cfg["dim"])
    loss_fn = functools.partial(LOSSES[cfg["loss"]], **cfg["loss_kwargs"])
    batch_loss = functools.partial(_align_batch_loss, tokenizer=tokenizer, texts=texts, targets=targets,
                                   loss_fn=loss_fn, cfg=cfg, device=device)
    run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, len(texts), batch_loss, data, "align")


def _align_batch_loss(model, global_idx, local, epoch, tokenizer, texts, targets, loss_fn, cfg, device):
    idx = global_idx[local]
    enc = tokenize(tokenizer, [texts[i] for i in idx], cfg["max_query_len"], cfg["query_prefix"])
    student = embed(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), cfg["dim"])
    return {"loss": loss_fn(student.float(), targets[idx].to(device))}


# ---------- shared by align_main and refine.refine_main ----------

def final_exists(out_dir):
    if os.path.exists(os.path.join(out_dir, "final")):
        if is_main():
            print(f"[train] {out_dir}/final exists, nothing to do")
        return True
    return False


def load_rank0_first(name, device):
    """load_encoder on every rank; rank 0 downloads first, the others then read the HF cache."""
    rank = get_rank()
    if not is_main():
        print(f"[rank{rank}] waiting for rank 0 to load the model", flush=True)
        barrier()
    print(f"[rank{rank}] loading {name}", flush=True)
    model, tokenizer = load_encoder(name, device)
    print(f"[rank{rank}] model loaded on {device}{gpu_mem(device)}", flush=True)
    if is_main():
        barrier()
    return model, tokenizer


def run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, n_items, batch_loss, data, tag):
    """The training loop over n_items examples, shared by the phases. Only the adapter trains (DDP over its params).
    batch_loss(ddp_model, global_idx, local, epoch) -> {"loss": tensor, other logged parts...}: global_idx is the
    whole global batch (same on every rank), global_idx[local] this rank's share. Each epoch is one seeded
    permutation; the ragged tail is dropped so every rank runs the same steps. Logs / evals / saves on rank 0;
    resumes from the latest step checkpoint; writes <out_dir>/final at the end.
    A step = grad_accum micro-batches of batch_size * world_size (gradients all-reduced on the last one only).
    patience > 0: <out_dir>/best holds the lowest-val_loss weights, training stops after patience evals without an
    improvement, and best becomes final."""
    body, head = trainable_params(model)  # LoRA + head, or every weight + head for a fully trained model
    params = body + head
    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    micro = bs * world_size
    per_step = micro * accum
    steps_per_epoch = n_items // per_step
    total = steps_per_epoch * cfg["epochs"]
    assert total > 0, f"{n_items} examples < one global batch of {per_step}"
    warmup = int(cfg["warmup_ratio"] * total)
    opt = torch.optim.AdamW([{"params": body, "lr": cfg["lr"]}, {"params": head, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, functools.partial(lr_factor, warmup=warmup, total=total))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best, bad = 0, float("inf"), 0  # best val_loss so far, evals since it improved (rank 0)
    best_dir = os.path.join(cfg["out_dir"], "best")
    ckpt = (step_checkpoints(cfg["out_dir"]) or [None])[-1]
    if ckpt:
        load_adapter_weights(model, ckpt)
        state = torch.load(os.path.join(ckpt, TRAINER_STATE), map_location=device)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, best, bad = state["step"], state.get("best", best), state.get("bad", bad)
    if is_main():
        n_params = sum(p.numel() for p in params)
        print(f"[{tag}] {n_items} examples, global batch {per_step}, {steps_per_epoch} steps/epoch, {total} steps, "
              f"{n_params / 1e6:.2f}M trainable params" + (f", resumed from {ckpt}" if ckpt else ""))
        val = load_val(data, cfg)

    model.train()
    print(f"[rank{rank}] wrapping in DDP{gpu_mem(device)}", flush=True)
    ddp_model = wrap_ddp(model, device)
    print(f"[rank{rank}] training from step {step} of {total}", flush=True)
    local = slice(rank * bs, (rank + 1) * bs)
    epoch, order, t0 = None, None, time.perf_counter()
    while step < total:
        if step // steps_per_epoch != epoch:
            epoch = step // steps_per_epoch
            order = epoch_order(n_items, cfg["seed"], epoch)
        start = (step % steps_per_epoch) * per_step
        opt.zero_grad(set_to_none=True)
        parts = {}  # mean over the micro-batches, detached, for logging
        for k in range(accum):
            with ddp_model.no_sync() if k < accum - 1 else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = batch_loss(ddp_model, order[start + k * micro:start + (k + 1) * micro], local, epoch)
                scaler.scale(out["loss"] / accum).backward()
            for name, v in out.items():
                parts[name] = parts.get(name, 0) + v.detach() / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % cfg["log_every"] == 0 or step == total:
            means = {("train_loss" if k == "loss" else k): reduce_mean(float(v), device) for k, v in parts.items()}
            if is_main():
                log_row(cfg, tag, dict(step=step, epoch=epoch, **means, lr=sched.get_last_lr()[0],
                                       min=(time.perf_counter() - t0) / 60))
        if (cfg["eval_every"] and step % cfg["eval_every"] == 0) or step == total:
            stop = 0.0
            if is_main():
                row = evaluate_val(model, tokenizer, val, cfg, device)
                log_row(cfg, tag, dict(step=step, **row))
                if cfg["patience"]:
                    if row["val_loss"] < best:
                        best, bad = row["val_loss"], 0
                        save_adapter(model, best_dir, adapter_cfg)
                        print(f"[train] saved {best_dir} (val_loss {best:.4g})")
                    else:
                        bad += 1
                    stop = float(bad >= cfg["patience"])
            else:
                print(f"[rank{rank}] step {step}: waiting for rank 0 to evaluate", flush=True)
            barrier()
            if device.type == "cuda":  # peak since the previous eval, train steps + (rank 0) this eval
                print(f"[rank{rank}] step {step}: GPU peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB "
                      f"allocated, {torch.cuda.max_memory_reserved(device) / 2**30:.1f} GB reserved{gpu_mem(device)}",
                      flush=True)
                torch.cuda.reset_peak_memory_stats(device)
            if reduce_mean(stop, device) > 0:  # every rank takes rank 0's decision
                if is_main():
                    print(f"[{tag}] early stop at step {step}: val_loss did not improve for {bad} evals "
                          f"(best {best:.4g})")
                break
        if cfg["save_every"] and step % cfg["save_every"] == 0 and step < total:
            if is_main():
                save_checkpoint(model, adapter_cfg, cfg, step, opt, sched, scaler, best, bad)
            else:
                print(f"[rank{rank}] step {step}: waiting for rank 0 to save", flush=True)
            barrier()
    if is_main():
        final = os.path.join(cfg["out_dir"], "final")
        if cfg["patience"] and os.path.exists(best_dir):  # missing only if no eval ever improved (e.g. NaN)
            os.replace(best_dir, final)
        else:
            save_adapter(model, final, adapter_cfg)
        print(f"[{tag}] done: {final}")
    print(f"[rank{rank}] training loop finished, waiting for all ranks", flush=True)
    barrier()


def load_val(data, cfg):
    """Rank 0's in-training eval set: val queries, their teacher vectors, the teacher corpus."""
    val_t, ids = load_emb(cfg["teacher_emb_dir"], "val")
    assert ids == data["val"]["ids"], "val: teacher vectors don't match data.json"
    docs, _ = load_emb(cfg["teacher_emb_dir"], "corpus")
    return dict(texts=data["val"]["texts"], pos=data["val"]["pos"], teacher=torch.from_numpy(val_t.astype(np.float32)),
                docs=docs)


def evaluate_val(model, tokenizer, val, cfg, device):
    """L2 + cosine to the teacher's val vectors, and retrieval from the teacher corpus."""
    q = encode(model, tokenizer, val["texts"], device, cfg["dim"], cfg["max_query_len"], cfg["query_prefix"],
               cfg["batch_size"]).float()
    model.train()  # encode switched it to eval
    m = score(q.numpy(), val["docs"], val["pos"], device=device)
    return {"val_loss": l2_loss(q, val["teacher"]).item(), "val_ndcg@10": m["ndcg@10"],
            "val_recall@10": m["recall@10"], "val_cos": (q * val["teacher"]).sum(-1).mean().item()}


def save_checkpoint(model, adapter_cfg, cfg, step, opt, sched, scaler, best, bad):
    path = os.path.join(cfg["out_dir"], f"step_{step:07d}")
    save_adapter(model, path, adapter_cfg, dict(step=step, opt=opt.state_dict(), sched=sched.state_dict(),
                                                 scaler=scaler.state_dict(), best=best, bad=bad))
    for p in step_checkpoints(cfg["out_dir"])[:-cfg["keep_last"]]:
        shutil.rmtree(p)
    print(f"[train] saved {path}")


def log_row(cfg, tag, row):
    """Print + append to <out_dir>/log.jsonl (rank 0)."""
    print(f"[{tag}] " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
          flush=True)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def read_log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl")) as f:
        return [json.loads(line) for line in f]

"""Student training: stage A (align to teacher query vectors) and stage C (contrastive refinement against the
frozen teacher doc vectors). Both are DDP workers: ddp.launch(align_main, cfg) / ddp.launch(cr_main, cfg).
Each stage reads data.json + teacher embeddings and writes out_dir/{weights, cfg.json, train_log.jsonl}."""
import gc
import json
import math
import os
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import load_emb, load_json, make_loader, resolve_cfg, save_json
from ddp import barrier, cleanup, get_rank, is_main, launch, reduce_mean, setup, wrap_ddp
from losses import LOSSES
from model import embed, load_encoder, param_counts, save_encoder

DEBUG_STEPS = 3
BENCH_WARMUP = 3

ALIGN_DEFAULTS = dict(
    init="Qwen/Qwen3-Embedding-0.6B",  # hub id, checkpoint dir, or "tiny:<seed>"
    data_path="cache/data.json",
    emb_dir="cache/emb/teacher",       # teacher "align" + "val" query vectors are the targets
    out_dir="runs/aligned",
    dim=256, max_query_len=32,
    loss="l2", loss_kwargs={},
    lr=2e-5, lr_schedule="cosine", warmup_ratio=0.05, min_lr_ratio=0.1,
    epochs=1, batch_size=128, accum_steps=1, max_grad_norm=1.0,
    trainable="all_but_embeddings",    # all | all_but_embeddings
    amp=True,                          # fp16 autocast + GradScaler (CUDA). False = plain fp32
    seed=0, val_every=200, save_dtype="fp16",
    debug=False, plan_only=False, max_steps=None, save=True, bench_file=None,
)

CR_DEFAULTS = dict(
    init="Qwen/Qwen3-Embedding-0.6B",
    data_path="cache/data.json",
    emb_dir="cache/emb/teacher",       # teacher "corpus" vectors are the frozen doc table
    out_dir="runs/cr_only",
    dim=256, max_query_len=32,
    loss="contrastive",
    loss_kwargs=dict(tau=0.05, n_hard_neg=1, same_tower=True, mask_margin=0.1, extra_negs=0),
    lr=1e-5, lr_schedule="cosine", warmup_ratio=0.05, min_lr_ratio=0.1,
    epochs=1, batch_size=128, accum_steps=1, max_grad_norm=1.0,
    trainable="all_but_embeddings",
    amp=True,
    seed=0, val_every=200, save_dtype="fp16",
    debug=False, plan_only=False, max_steps=None, save=True, bench_file=None,
)


# ---------- lr schedules ----------

def cosine_schedule(optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.1):
    """Linear warmup to the optimizer's lr, then cosine decay to lr * min_lr_ratio at total_steps (optimizer updates)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def constant_schedule(optimizer, total_steps, warmup_steps=0, min_lr_ratio=None):
    return LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))


LR_SCHEDULES = {"cosine": cosine_schedule, "constant": constant_schedule}


def num_update_steps(n_batches, accum_steps, epochs):
    return math.ceil(n_batches / accum_steps) * epochs


def _plan_batches(cfg, n_batches):
    """(batches per epoch, accum, epochs) actually run, honouring debug and max_steps."""
    if cfg["debug"]:
        return min(n_batches, DEBUG_STEPS), 1, 1
    if cfg["max_steps"]:
        return min(n_batches, cfg["max_steps"] * cfg["accum_steps"]), cfg["accum_steps"], 1
    return n_batches, cfg["accum_steps"], cfg["epochs"]


# ---------- the loop ----------

def train_loop(model, loader, optimizer, scheduler, device, cfg, prepare, val_fn=None):
    """batch -> forward -> LOSSES[cfg["loss"]] -> backward -> step, for all epochs.
    Grad accumulation with no_sync on non-update steps; fp16 autocast + GradScaler when cfg["amp"] on CUDA.
    Logs (reduced across ranks) every val_every updates and at the end. Returns the log."""
    loss_fn = LOSSES[cfg["loss"]]
    n_batches, accum, epochs = _plan_batches(cfg, len(loader))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    params = [p for p in model.parameters() if p.requires_grad]
    log, running, step = [], [], 0
    model.train()
    for epoch in range(epochs):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        bar = tqdm(loader, total=n_batches, disable=cfg["debug"] or not is_main(), desc=f"epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(bar):
            if i >= n_batches:
                break
            first = cfg["debug"] and epoch == 0 and i == 0
            batch = prepare(batch, device)
            update = (i + 1) % accum == 0 or i + 1 == n_batches
            sync = nullcontext() if update or not isinstance(model, DDP) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    q = embed(model, batch["input_ids"], batch["attention_mask"], cfg["dim"])
                loss = loss_fn(q.float(), batch, **cfg["loss_kwargs"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {loss.item()} (if amp=True, retry with amp=False)")
                scaler.scale(loss / accum).backward()
            if first and is_main():
                print_shapes(q=q, **{k: v for k, v in batch.items() if torch.is_tensor(v)})
            if update:
                scaler.unscale_(optimizer)
                if first:
                    check_grads(model, finite=not use_amp)  # with amp, GradScaler skips steps with inf grads
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            running.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if update and (step % cfg["val_every"] == 0 or i + 1 == n_batches):
                rec = {"step": step, "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                       "train_loss": reduce_mean(float(np.mean(running)), device)}
                running = []
                if val_fn is not None:
                    rec.update(val_fn(model))
                    model.train()
                log.append(rec)
                if is_main():
                    print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                 for k, v in rec.items()))
    return log


@torch.no_grad()
def val_metrics(model, loader, device, cfg, prepare):
    """Mean stage loss and mean cos(student query, teacher query) on the val queries, averaged over ranks."""
    model.eval()
    loss_fn = LOSSES[cfg["loss"]]
    n = min(len(loader), DEBUG_STEPS) if cfg["debug"] else len(loader)
    losses, cos = [], []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        batch = prepare(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
            q = embed(model, batch["input_ids"], batch["attention_mask"], cfg["dim"]).float()
        losses.append(loss_fn(q, batch, **cfg["loss_kwargs"]).item())
        cos.append((q * batch["target"]).sum(-1).mean().item())
    return {"val_loss": reduce_mean(float(np.mean(losses)), device),
            "val_cos_to_teacher": reduce_mean(float(np.mean(cos)), device)}


# ---------- stage plumbing ----------

def run_stage(rank, world_size, cfg, make_data, prepare):
    """Shared worker body: setup -> model -> loaders -> optimizer/schedule -> plan -> train -> save (rank 0).
    make_data(cfg, tokenizer, device) -> (train_loader, val_loader, ctx); prepare(batch, device, ctx, cfg) -> batch."""
    device = setup(rank, world_size, timeout_s=120 if cfg["debug"] else 1800)
    try:
        torch.manual_seed(cfg["seed"])
        if is_main():
            print("[cfg] " + json.dumps(cfg, indent=1, default=str))
        if not is_main():
            barrier()  # rank 0 downloads first
        model, tokenizer = load_encoder(cfg["init"], device, trainable=cfg["trainable"])
        if is_main():
            barrier()
        train_loader, val_loader, ctx = make_data(cfg, tokenizer, device)
        prep = lambda batch, dev: prepare(batch, dev, ctx, cfg)  # noqa: E731
        model = wrap_ddp(model, device)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
        n_batches, accum, epochs = _plan_batches(cfg, len(train_loader))
        total = num_update_steps(n_batches, accum, epochs)
        scheduler = LR_SCHEDULES[cfg["lr_schedule"]](optimizer, total, int(cfg["warmup_ratio"] * total), cfg["min_lr_ratio"])
        if is_main():
            print_plan(model, cfg, world_size, train_loader, val_loader, total)
        if cfg["plan_only"]:
            if is_main():
                print("[plan] first batch:")
                print_shapes(**{k: v for k, v in prep(next(iter(train_loader)), device).items() if torch.is_tensor(v)})
            return
        if cfg["bench_file"]:
            _benchmark_run(model, train_loader, optimizer, scheduler, device, cfg, prep, world_size)
            return
        val_fn = lambda m: val_metrics(m, val_loader, device, cfg, prep)  # noqa: E731
        log = train_loop(model, train_loader, optimizer, scheduler, device, cfg, prep, val_fn)
        if cfg["debug"]:
            assert_ranks_in_sync(model, device)
        if is_main() and cfg["save"]:
            save_encoder(model, tokenizer, cfg["out_dir"], cfg["save_dtype"])
            save_json(cfg, os.path.join(cfg["out_dir"], "cfg.json"))
            with open(os.path.join(cfg["out_dir"], "train_log.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in log)
            print(f"[train] saved to {cfg['out_dir']}")
        barrier()
    finally:
        cleanup()


def _teacher(emb_dir, name, dim):
    emb, ids = load_emb(emb_dir, name)
    if emb.shape[1] != dim:
        raise ValueError(f"teacher {name} vectors have dim {emb.shape[1]} but cfg dim={dim}; re-embed or change dim")
    return emb, ids


def _check_ids(name, got, want):
    if got != want:
        raise ValueError(f"teacher {name} ids do not match data.json; re-run the teacher cell of 00")


def align_data(cfg, tokenizer, device):
    data = load_json(cfg["data_path"])
    loaders = []
    for name, shuffle in (("align", True), ("val", False)):
        emb, ids = _teacher(cfg["emb_dir"], name, cfg["dim"])
        _check_ids(name, ids, data[name]["ids"])
        items = [{"text": t, "target": emb[i]} for i, t in enumerate(data[name]["texts"])]
        loaders.append(make_loader(items, tokenizer, cfg["batch_size"], cfg["max_query_len"], shuffle=shuffle,
                                   distributed=True, seed=cfg["seed"], drop_last=shuffle))
    return loaders[0], loaders[1], None


def prepare_align(batch, device, ctx, cfg):
    return {k: v.to(device) for k, v in batch.items()}


def cr_data(cfg, tokenizer, device):
    """Train items = CR triples (query, positive, hard negative) + the teacher query vector (for hybrid losses and
    val cosine). ctx holds the frozen teacher doc table on this rank's device, loaded once."""
    data = load_json(cfg["data_path"])
    docs, doc_ids = _teacher(cfg["emb_dir"], "corpus", cfg["dim"])
    _check_ids("corpus", doc_ids, data["corpus"]["ids"])
    align_emb, align_ids = _teacher(cfg["emb_dir"], "align", cfg["dim"])
    row = {qid: i for i, qid in enumerate(align_ids)}
    val_emb, _ = _teacher(cfg["emb_dir"], "val", cfg["dim"])
    targets = {"train": [align_emb[row[qid]] for qid in data["train"]["ids"]], "val": list(val_emb)}
    loaders = []
    for name, shuffle in (("train", True), ("val", False)):
        s = data[name]
        items = [{"text": t, "pos": p[0], "neg": n, "target": e}
                 for t, p, n, e in zip(s["texts"], s["pos"], s["neg"], targets[name])]
        loaders.append(make_loader(items, tokenizer, cfg["batch_size"], cfg["max_query_len"], shuffle=shuffle,
                                   distributed=True, seed=cfg["seed"], drop_last=shuffle))
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + get_rank())
    return loaders[0], loaders[1], {"docs": torch.from_numpy(docs).to(device), "gen": gen}


def prepare_cr(batch, device, ctx, cfg):
    """Look up teacher doc vectors for the batch; sample extra random corpus negatives if asked."""
    batch = {k: v.to(device) for k, v in batch.items()}
    docs = ctx["docs"]
    batch["pos_emb"], batch["neg_emb"] = docs[batch["pos"]].float(), docs[batch["neg"]].float()
    n_extra = cfg["loss_kwargs"].get("extra_negs", 0)
    if n_extra:
        batch["extra"] = torch.randint(len(docs), (n_extra,), device=device, generator=ctx["gen"])
        batch["extra_emb"] = docs[batch["extra"]].float()
    return batch


def align_main(rank, world_size, cfg):
    """Stage A worker: ddp.launch(align_main, cfg). Trains on cached teacher query vectors (no teacher forward)."""
    run_stage(rank, world_size, resolve_cfg(ALIGN_DEFAULTS, cfg), align_data, prepare_align)


def cr_main(rank, world_size, cfg):
    """Stage C worker: ddp.launch(cr_main, cfg). init can be the pretrained student or an aligned checkpoint dir."""
    run_stage(rank, world_size, resolve_cfg(CR_DEFAULTS, cfg), cr_data, prepare_cr)


# ---------- scaling benchmark ----------

def _benchmark_run(model, loader, optimizer, scheduler, device, cfg, prep, world_size):
    train_loop(model, loader, optimizer, scheduler, device, {**cfg, "max_steps": BENCH_WARMUP, "val_every": 10**9}, prep)
    _sync(device)
    t0 = time.perf_counter()
    train_loop(model, loader, optimizer, scheduler, device, {**cfg, "val_every": 10**9}, prep)
    _sync(device)
    seconds = reduce_mean(time.perf_counter() - t0, device)
    if is_main():
        samples = cfg["max_steps"] * cfg["accum_steps"] * cfg["batch_size"] * world_size
        save_json({"world_size": world_size, "samples": samples, "seconds": seconds}, cfg["bench_file"])
    barrier()


def benchmark_scaling(worker, cfg, steps=20, nprocs_list=(1, 2)):
    """Run `worker` (align_main / cr_main) for `steps` timed updates (after BENCH_WARMUP discarded ones) with
    each process count; print samples/s and scaling efficiency. Writes no checkpoints."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in nprocs_list:
            bench_file = os.path.join(tmp, f"bench{n}.json")
            launch(worker, {**cfg, "max_steps": steps, "save": False, "debug": False, "bench_file": bench_file}, nprocs=n)
            r = load_json(bench_file)
            results[n] = r["samples"] / r["seconds"]
    base = results[nprocs_list[0]] / nprocs_list[0]
    for n, sps in results.items():
        print(f"[scaling] nprocs={n}: {sps:,.1f} samples/s  efficiency={sps / (n * base):.0%}")
    return results


# ---------- checks and utils ----------

def print_plan(model, cfg, world_size, train_loader, val_loader, total_updates):
    trainable, n_params = param_counts(model)
    print(f"[plan] world_size={world_size} per-GPU batch={cfg['batch_size']} accum={cfg['accum_steps']} "
          f"effective batch={cfg['batch_size'] * world_size * cfg['accum_steps']}")
    print(f"[plan] train batches/rank/epoch={len(train_loader)} val batches/rank={len(val_loader)} "
          f"optimizer updates={total_updates}")
    print(f"[plan] trainable params={trainable:,} / {n_params:,} ({trainable / n_params:.1%})")


def print_shapes(**tensors):
    for name, t in tensors.items():
        print(f"  {name:<15} {tuple(t.shape)}  {t.dtype}  {t.device}")


def check_grads(model, finite=True):
    """Every trainable param must get a gradient (and a finite one, unless amp may legitimately overflow)."""
    missing, bad = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif finite and not torch.isfinite(p.grad).all():
            bad.append(name)
    if missing or bad:
        raise RuntimeError(f"grad check failed. no grad: {missing[:5]} non-finite: {bad[:5]}")
    if is_main():
        print(f"  grads OK for {sum(p.requires_grad for p in model.parameters())} trainable tensors")


@torch.no_grad()
def assert_ranks_in_sync(model, device):
    """DDP sanity check: every trainable weight must be identical on all ranks."""
    if not torch.distributed.is_initialized():
        return
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        hi, lo = p.detach().clone(), p.detach().clone()
        torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN)
        if not torch.equal(hi, lo):
            raise RuntimeError(f"weights diverged across ranks: {name}")
    if is_main():
        print("  ranks in sync")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def clean_unused_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

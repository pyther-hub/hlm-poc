"""Alignment fine-tuning: a frozen student (Qwen3-Embedding-0.6B) + adapter (LoRA on every decoder linear, linear
head to the teacher dim) trained so its query vectors match the teacher's cached query vectors (emb/teacher/align.npy).
DDP: one process per GPU, each takes its slice of every global batch; only the adapter has gradients.
Checkpoints are adapter-only (model.save_adapter): <out_dir>/step_<n>/ (+ trainer_state.pt, resumable) and
<out_dir>/final/, which load_encoder / evaluate_model accept like a model dir."""
import functools
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import torch

from data import load_emb, load_json, resolve_cfg
from ddp import barrier, cleanup, is_main, reduce_mean, setup, wrap_ddp
from evaluation import score
from model import (TRAINER_STATE, adapter_params, attach_adapter, embed, encode, load_adapter_weights, load_encoder,
                   save_adapter, tokenize)

ALIGN_DEFAULTS = dict(
    model="Qwen/Qwen3-Embedding-0.6B",  # student base: hub id or local dir; frozen, fp32 (amp does the speed-up)
    dim=256,                            # head output = teacher index dim
    lora_rank=16,
    lora_alpha=32,                      # scale = alpha / rank
    lora_dropout=0.05,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    query_prefix="",                    # student query prompt
    max_query_len=32,
    loss="l2",                          # name in LOSSES
    loss_kwargs={},
    lr=2e-4,                            # LoRA params
    head_lr=1e-3,                       # head params
    weight_decay=0.0,
    warmup_ratio=0.05,                  # linear warmup, then cosine decay to 0
    epochs=1,
    batch_size=128,                     # per GPU; global batch = batch_size * world_size
    grad_clip=1.0,
    amp=True,                           # fp16 autocast + GradScaler on CUDA (T4 has no bf16)
    seed=0,
    log_every=20,                       # steps; train loss averaged over ranks
    eval_every=200,                     # steps; val loss + retrieval on rank 0, 0 = only at the end
    save_every=500,                     # steps; resumable adapter checkpoint
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
    try:
        if os.path.exists(os.path.join(cfg["out_dir"], "final")):
            if is_main():
                print(f"[align] {cfg['out_dir']}/final exists, nothing to do")
            return
        train(rank, world_size, cfg, device)
    finally:
        cleanup()


def train(rank, world_size, cfg, device):
    data = load_json(cfg["data_path"])
    texts = data["align"]["texts"]
    targets, ids = load_emb(cfg["teacher_emb_dir"], "align")
    assert ids == data["align"]["ids"], "align: teacher vectors don't match data.json"
    assert targets.shape[1] == cfg["dim"], f"teacher dim {targets.shape[1]} != cfg dim {cfg['dim']}"
    targets = torch.from_numpy(targets.astype(np.float32))

    if not is_main():
        barrier()  # rank 0 downloads first, the others read the HF cache
    model, tokenizer = load_encoder(cfg["model"], device)
    if is_main():
        barrier()
    torch.manual_seed(cfg["seed"])  # same adapter init on every rank (DDP also broadcasts it from rank 0)
    attach_adapter(model, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["lora_targets"], cfg["dim"])
    adapter_cfg = dict(base=cfg["model"], rank=cfg["lora_rank"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"],
                       targets=list(cfg["lora_targets"]), out_dim=cfg["dim"])
    lora, head = adapter_params(model)
    params = lora + head

    per_step = cfg["batch_size"] * world_size
    steps_per_epoch = len(texts) // per_step  # drop the ragged tail: every rank runs the same number of steps
    total = steps_per_epoch * cfg["epochs"]
    assert total > 0, f"{len(texts)} align queries < one global batch of {per_step}"
    warmup = int(cfg["warmup_ratio"] * total)
    opt = torch.optim.AdamW([{"params": lora, "lr": cfg["lr"]}, {"params": head, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, functools.partial(lr_factor, warmup=warmup, total=total))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = functools.partial(LOSSES[cfg["loss"]], **cfg["loss_kwargs"])

    step = 0
    ckpt = (step_checkpoints(cfg["out_dir"]) or [None])[-1]
    if ckpt:
        load_adapter_weights(model, ckpt)
        state = torch.load(os.path.join(ckpt, TRAINER_STATE), map_location=device)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
    if is_main():
        n_params = sum(p.numel() for p in params)
        print(f"[align] {len(texts)} queries, global batch {per_step}, {steps_per_epoch} steps/epoch, {total} steps, "
              f"{n_params / 1e6:.2f}M trainable params" + (f", resumed from {ckpt}" if ckpt else ""))
        val = _load_val(data, cfg)

    model.train()
    ddp_model = wrap_ddp(model, device)
    epoch, order, t0 = None, None, time.perf_counter()
    while step < total:
        if step // steps_per_epoch != epoch:
            epoch = step // steps_per_epoch
            order = epoch_order(len(texts), cfg["seed"], epoch)
        start = (step % steps_per_epoch) * per_step + rank * cfg["batch_size"]
        idx = order[start:start + cfg["batch_size"]]
        enc = tokenize(tokenizer, [texts[i] for i in idx], cfg["max_query_len"], cfg["query_prefix"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            student = embed(ddp_model, enc["input_ids"].to(device), enc["attention_mask"].to(device), cfg["dim"])
        loss = loss_fn(student.float(), targets[idx].to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % cfg["log_every"] == 0 or step == total:
            train_loss = reduce_mean(loss.item(), device)
            if is_main():
                _log(cfg, dict(step=step, epoch=epoch, train_loss=train_loss, lr=sched.get_last_lr()[0],
                               min=(time.perf_counter() - t0) / 60))
        if (cfg["eval_every"] and step % cfg["eval_every"] == 0) or step == total:
            if is_main():
                _log(cfg, dict(step=step, **_evaluate(model, tokenizer, val, cfg, device, loss_fn)))
            barrier()
        if step % cfg["save_every"] == 0 and step < total:
            if is_main():
                _save_checkpoint(model, adapter_cfg, cfg, step, opt, sched, scaler)
            barrier()
    if is_main():
        save_adapter(model, os.path.join(cfg["out_dir"], "final"), adapter_cfg)
        print(f"[align] done: {cfg['out_dir']}/final")
    barrier()


def _load_val(data, cfg):
    """Rank 0's in-training eval set: val queries, their teacher vectors, the teacher corpus."""
    val_t, ids = load_emb(cfg["teacher_emb_dir"], "val")
    assert ids == data["val"]["ids"], "val: teacher vectors don't match data.json"
    docs, _ = load_emb(cfg["teacher_emb_dir"], "corpus")
    return dict(texts=data["val"]["texts"], pos=data["val"]["pos"], teacher=torch.from_numpy(val_t.astype(np.float32)),
                docs=docs)


def _evaluate(model, tokenizer, val, cfg, device, loss_fn):
    """Val loss against the teacher's val vectors + retrieval from the teacher corpus."""
    q = encode(model, tokenizer, val["texts"], device, cfg["dim"], cfg["max_query_len"], cfg["query_prefix"],
               cfg["batch_size"]).float()
    model.train()  # encode switched it to eval
    m = score(q.numpy(), val["docs"], val["pos"], device=device)
    return {"val_loss": loss_fn(q, val["teacher"]).item(), "val_ndcg@10": m["ndcg@10"],
            "val_recall@10": m["recall@10"], "val_cos": (q * val["teacher"]).sum(-1).mean().item()}


def _save_checkpoint(model, adapter_cfg, cfg, step, opt, sched, scaler):
    path = os.path.join(cfg["out_dir"], f"step_{step:07d}")
    save_adapter(model, path, adapter_cfg, dict(step=step, opt=opt.state_dict(), sched=sched.state_dict(),
                                                 scaler=scaler.state_dict()))
    for p in step_checkpoints(cfg["out_dir"])[:-cfg["keep_last"]]:
        shutil.rmtree(p)
    print(f"[align] saved {path}")


def _log(cfg, row):
    """Print + append to <out_dir>/log.jsonl (rank 0)."""
    print("[align] " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
          flush=True)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def read_log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl")) as f:
        return [json.loads(line) for line in f]

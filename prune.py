"""Pruning: cut a student to fewer decoder layers and a narrower FFN, then fully fine-tune it (every weight + head) on
the same L2 alignment as phase 2. The start point is an adapter dir: a phase-2 LoRA adapter (LoRA merged into the
base first) or a previous pruned stage (progressive pruning = chaining stages). Kept layers are evenly spaced (first
and last always kept); kept FFN neurons are the largest by |gate_j| * |up_j| * |down_:,j|. Checkpoints use the adapter
format with full=True plus keep_layers / ffn_idx (indices into the hub base), so load_encoder rebuilds the pruned
model from the base id. Same loop, checkpoints and resume as align.py (align.run)."""
import functools
import os
import traceback

import numpy as np
import torch

from align import _align_batch_loss, final_exists, l2_loss, load_rank0_first, run
from data import load_emb, load_json, resolve_cfg
from ddp import cleanup, setup
from model import ADAPTER_CONFIG, attach_head, gpu_mem, load_adapter_weights, merge_lora, prune_model

PRUNE_DEFAULTS = dict(
    init="runs/align/final",            # phase-2 adapter or previous pruned stage; read only
    n_layers=14,                        # decoder layers kept, evenly spaced over the init's layers
    ffn_keep=0.75,                      # fraction of the init's FFN width kept in every layer
    dim=256,                            # must equal the init's head output and the teacher index dim
    query_prefix="",                    # must match phase 2
    max_query_len=32,
    lr=2e-5,                            # all base weights (full fine-tune)
    head_lr=1e-4,                       # head params
    weight_decay=0.0,
    warmup_ratio=0.05,                  # linear warmup, then cosine decay to 0
    epochs=1,
    batch_size=64,                      # per GPU; global batch = batch_size * world_size * grad_accum
    grad_accum=1,                       # micro-batches per optimizer step
    grad_clip=1.0,
    amp=True,                           # fp16 autocast + GradScaler on CUDA
    seed=0,
    log_every=20,
    eval_every=200,                     # val L2/cos + val retrieval on rank 0, 0 = only at the end
    patience=0,                         # evals without a val_loss drop before stopping, best weights -> final; 0 = off
    save_every=500,                     # 0 = no resumable step checkpoints
    keep_last=2,
    data_path="cache/data.json",
    teacher_emb_dir="cache/emb/teacher",
    out_dir="runs/prune",
)


def select_layers(n, k):
    """k evenly spaced indices into n layers, first and last included."""
    idx = sorted(set(np.round(np.linspace(0, n - 1, k)).astype(int).tolist()))
    assert len(idx) == k, f"can't pick {k} distinct layers out of {n}"
    return idx


def ffn_scores(mlp):
    """Per intermediate neuron j: |gate_j| * |up_j| * |down_:,j| (row / column L2 norms)."""
    w = [mlp.gate_proj.weight.float(), mlp.up_proj.weight.float(), mlp.down_proj.weight.float()]
    return w[0].norm(dim=1) * w[1].norm(dim=1) * w[2].norm(dim=0)


def top_neurons(mlp, k):
    return sorted(ffn_scores(mlp).topk(k).indices.tolist())


def pruned_student(init, n_layers, ffn_keep, device):
    """(model, tokenizer, adapter_cfg): the init adapter's model as plain weights + head, cut to n_layers and
    ffn_keep of its FFN width. adapter_cfg records the kept layers / neurons relative to the hub base."""
    init_cfg = load_json(os.path.join(init, ADAPTER_CONFIG))
    model, tokenizer = load_rank0_first(init_cfg["base"], device)
    keep = init_cfg.get("keep_layers", list(range(model.config.num_hidden_layers)))
    ffn = init_cfg.get("ffn_idx", [list(range(model.config.intermediate_size))] * len(keep))
    if init_cfg.get("full"):  # a previous pruned stage
        prune_model(model, keep, ffn)
        attach_head(model, init_cfg["out_dim"])
        load_adapter_weights(model, init)
    else:  # a LoRA adapter on the unpruned base
        merge_lora(model, init)
    sel = select_layers(len(keep), n_layers)
    neurons = [top_neurons(model.layers[i].mlp, round(ffn_keep * len(ffn[i]))) for i in sel]
    prune_model(model, sel, neurons)
    adapter_cfg = dict(base=init_cfg["base"], out_dim=init_cfg["out_dim"], full=True,
                       keep_layers=[keep[i] for i in sel],
                       ffn_idx=[[ffn[i][j] for j in n] for i, n in zip(sel, neurons)])
    return model, tokenizer, adapter_cfg


def prune_main(rank, world_size, cfg):
    """One worker of ddp.launch(prune_main, cfg). Resumes from the latest step checkpoint; no-op if final/ exists."""
    cfg = resolve_cfg(PRUNE_DEFAULTS, cfg)
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
    targets, ids = load_emb(cfg["teacher_emb_dir"], "align")
    assert ids == data["align"]["ids"], "align: teacher vectors don't match data.json"
    model, tokenizer, adapter_cfg = pruned_student(cfg["init"], cfg["n_layers"], cfg["ffn_keep"], device)
    print(f"[rank{rank}] pruned to {len(adapter_cfg['keep_layers'])} layers{gpu_mem(device)}", flush=True)
    assert adapter_cfg["out_dim"] == cfg["dim"] == targets.shape[1], \
        f"init dim {adapter_cfg['out_dim']}, cfg dim {cfg['dim']}, teacher dim {targets.shape[1]}"
    model.requires_grad_(True)
    batch_loss = functools.partial(_align_batch_loss, tokenizer=tokenizer, texts=texts,
                                   targets=torch.from_numpy(targets.astype(np.float32)), loss_fn=l2_loss, cfg=cfg,
                                   device=device)
    run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, len(texts), batch_loss, data, "prune")
       
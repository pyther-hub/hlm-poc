"""Pruning: cut a student to fewer decoder layers and a narrower FFN, then fully fine-tune it (every weight + head) on
the same L2 alignment as phase 2. The start point is an adapter dir: a phase-2 LoRA adapter (LoRA merged into the
base first) or a previous pruned stage (progressive pruning = chaining stages). Kept layers are evenly spaced (first
and last always kept); kept FFN neurons are the largest by |gate_j| * |up_j| * |down_:,j|. Checkpoints use the adapter
format with full=True plus keep_layers / ffn_idx (indices into the hub base), so load_encoder rebuilds the pruned
model from the base id. Same loop, checkpoints and resume as align.py (align.run).
layer_score="l2" keeps the layers chosen greedily by calibration L2 instead (l2_layers); rank 0 picks them and writes
<out_dir>/prune_plan.json, every rank reads it."""
import contextlib
import functools
import os
import traceback

import numpy as np
import torch

from align import _align_batch_loss, final_exists, l2_loss, load_rank0_first, run
from data import load_emb, load_json, resolve_cfg, save_json
from ddp import barrier, cleanup, is_main, setup
from model import ADAPTER_CONFIG, attach_head, embed, gpu_mem, load_adapter_weights, merge_lora, prune_model, tokenize

PRUNE_DEFAULTS = dict(
    init="runs/align/final",            # phase-2 adapter or previous pruned stage; read only
    n_layers=14,                        # decoder layers kept from the init's layers
    layer_score="even",                 # "even": evenly spaced; "l2": greedy, drop the layer whose removal raises calib L2 least
    calib_size=1024,                    # layer_score="l2": align queries (seeded sample) the L2 is measured on
    ffn_keep=0.75,                      # fraction of the init's FFN width kept in every layer
    freeze_embeddings=False,            # True: token embeddings stay as in init (no grads / AdamW state: -12 B/param)
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


@contextlib.contextmanager
def only_layers(model, keep):
    """Run model with only model.layers[keep] inside the block (like prune_model, undone on exit). Qwen3Model loops
    over layers[:config.num_hidden_layers] with config.layer_types[i]; layer_idx indexes the KV cache."""
    layers, config = model.layers, model.config
    n, types, idx = config.num_hidden_layers, getattr(config, "layer_types", None), [l.self_attn.layer_idx for l in layers]
    model.layers = torch.nn.ModuleList([layers[i] for i in keep])
    for j, layer in enumerate(model.layers):
        layer.self_attn.layer_idx = j
    config.num_hidden_layers = len(keep)
    if types:
        config.layer_types = [types[i] for i in keep]
    try:
        yield
    finally:
        model.layers, config.num_hidden_layers = layers, n
        if types:
            config.layer_types = types
        for layer, i in zip(layers, idx):
            layer.self_attn.layer_idx = i


@torch.no_grad()
def calib_l2(model, batches, dim):
    """Mean L2 to the teacher over batches of (input_ids, attention_mask, teacher vectors), fp16 autocast on CUDA."""
    total, n = 0.0, 0
    for ids, mask, target in batches:
        with torch.autocast("cuda", dtype=torch.float16, enabled=ids.is_cuda):
            q = embed(model, ids, mask, dim).float()
        total += l2_loss(q, target).item() * len(ids)
        n += len(ids)
    return total / n


def l2_layers(model, n, batches, dim):
    """Greedy L2-aware selection: repeatedly drop the layer whose removal raises calib_l2 least (every remaining layer
    re-scored after each drop) until n are left. Returns (kept indices, dropped in order, calib L2 after each drop;
    curve[0] = no layer dropped)."""
    keep, dropped, curve = list(range(len(model.layers))), [], [calib_l2(model, batches, dim)]
    while len(keep) > n:
        losses = []
        for i in keep:
            with only_layers(model, [k for k in keep if k != i]):
                losses.append(calib_l2(model, batches, dim))
        j = int(np.argmin(losses))
        dropped.append(keep.pop(j))
        curve.append(losses[j])
        print(f"[prune] dropped layer {dropped[-1]:>2}: calib L2 {curve[-1]:.4f}, {len(keep)} left", flush=True)
    return keep, dropped, curve


def calib_batches(cfg, tokenizer, texts, targets, device):
    """cfg["calib_size"] align queries (seeded by cfg["seed"]) as tokenized batches of cfg["batch_size"] on device."""
    idx = np.sort(np.random.default_rng(cfg["seed"]).permutation(len(texts))[:cfg["calib_size"]])
    out = []
    for start in range(0, len(idx), cfg["batch_size"]):
        part = idx[start:start + cfg["batch_size"]]
        enc = tokenize(tokenizer, [texts[i] for i in part], cfg["max_query_len"], cfg["query_prefix"])
        out.append((enc["input_ids"].to(device), enc["attention_mask"].to(device),
                    torch.from_numpy(targets[part].astype(np.float32)).to(device)))
    return out


def planned_layers(model, tokenizer, cfg, texts, targets, device):
    """layer_score="l2": rank 0 runs l2_layers (unless <out_dir>/prune_plan.json exists) and writes the plan; every
    rank then reads it, so all ranks keep the same layers. Indices are into the init's layers."""
    path = os.path.join(cfg["out_dir"], "prune_plan.json")
    if is_main() and not os.path.exists(path):
        model.eval()
        keep, dropped, curve = l2_layers(model, cfg["n_layers"],
                                         calib_batches(cfg, tokenizer, texts, targets, device), cfg["dim"])
        save_json(dict(keep=keep, dropped=dropped, calib_l2=curve), path)
    barrier()
    return load_json(path)["keep"]


def pruned_student(init, n_layers, ffn_keep, device, choose_layers=None):
    """(model, tokenizer, adapter_cfg): the init adapter's model as plain weights + head, cut to n_layers and
    ffn_keep of its FFN width. Layers are evenly spaced, or choose_layers(model, tokenizer) -> indices into the init's
    layers. adapter_cfg records the kept layers / neurons relative to the hub base."""
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
    sel = choose_layers(model, tokenizer) if choose_layers else select_layers(len(keep), n_layers)
    assert len(sel) == n_layers, f"chose {len(sel)} layers, want {n_layers}"
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
    assert cfg["layer_score"] in ("even", "l2"), cfg["layer_score"]
    choose = None
    if cfg["layer_score"] == "l2":
        choose = functools.partial(planned_layers, cfg=cfg, texts=texts, targets=targets, device=device)
    model, tokenizer, adapter_cfg = pruned_student(cfg["init"], cfg["n_layers"], cfg["ffn_keep"], device, choose)
    print(f"[rank{rank}] pruned to {len(adapter_cfg['keep_layers'])} layers{gpu_mem(device)}", flush=True)
    assert adapter_cfg["out_dim"] == cfg["dim"] == targets.shape[1], \
        f"init dim {adapter_cfg['out_dim']}, cfg dim {cfg['dim']}, teacher dim {targets.shape[1]}"
    model.requires_grad_(True)
    if cfg["freeze_embeddings"]:  # Qwen3: 151k x 1024 = 155M of the 596M params
        model.embed_tokens.requires_grad_(False)
    batch_loss = functools.partial(_align_batch_loss, tokenizer=tokenizer, texts=texts,
                                   targets=torch.from_numpy(targets.astype(np.float32)), loss_fn=l2_loss, cfg=cfg,
                                   device=device)
    run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, len(texts), batch_loss, data, "prune")
       
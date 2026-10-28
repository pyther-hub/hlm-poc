"""Contrastive refinement: continue training the phase-2 adapter (LoRA + head; base frozen) with
L_CR = -1/B sum_i log exp(s(q_i, d_i+)/tau) / (exp(s(q_i, d_i+)/tau) + sum_{u in N_i} exp(s_u/tau)),
s(q, d) = <student(q), teacher(d)>, documents = the frozen teacher corpus vectors (corpus.npy, never re-encoded),
plus l2_weight * L2(student q, teacher q) to stay near the teacher query space.
N_i = every doc of the global batch other than d_i+ (each query's positive + its n_neg negatives: the stored
negative[0] plus n_neg - 1 mined by teacher score, see mine_negatives; the global batch
is the same on every rank, so these are free) + the other student queries on this rank (same tower), minus false
negatives: q_i's other labelled positives and anything scoring above s(q_i, d_i+) + margin.
Same loop, checkpoints and resume as align.py (align.run)."""
import functools
import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from align import final_exists, l2_loss, load_rank0_first, run
from data import load_emb, load_json, resolve_cfg
from ddp import cleanup, setup
from model import ADAPTER_CONFIG, adapter_params, embed, tokenize

REFINE_DEFAULTS = dict(
    init_adapter="runs/align/final",    # phase-2 adapter dir (LoRA + head trained further) or a pruned one (all weights)
    dim=256,                            # must equal the adapter's head output and the teacher index dim
    query_prefix="",                    # must match phase 2
    max_query_len=32,
    tau=0.05,                           # temperature
    margin=0.1,                         # negative with s > s(q, d+) + margin is masked as a false negative
    n_neg=1,                            # negatives per query: 1 = the stored one, k > 1 adds k-1 teacher-mined ones
    l2_weight=0.1,                      # weight of L2(student q, teacher q); 0 = pure L_CR
    lr=5e-5,                            # LoRA params
    head_lr=1e-4,                       # head params
    weight_decay=0.0,
    warmup_ratio=0.05,                  # linear warmup, then cosine decay to 0
    epochs=1,
    batch_size=64,                      # queries per GPU; docs per step = 2 * batch_size * world_size
    grad_clip=1.0,
    amp=True,                           # fp16 autocast + GradScaler on CUDA
    seed=0,
    log_every=20,
    eval_every=200,                     # val L2/cos + val retrieval on rank 0, 0 = only at the end
    save_every=500,
    keep_last=2,
    data_path="cache/data.json",
    teacher_emb_dir="cache/emb/teacher",
    out_dir="runs/refine",
)


def contrastive_loss(q, docs, pos_col, known_pos, tau, margin):
    """L_CR for B student queries q [B, d] against M teacher docs [M, d] (both unit norm).
    pos_col [B]: column of each query's positive. known_pos [B, M] bool: other columns that are labelled positives of
    that query. Negatives = every other doc column + the other queries in q. Masked (false negatives): known_pos,
    and any negative scoring above s(q_i, d_i+) + margin. Returns (loss, fraction of negatives masked)."""
    rows = torch.arange(len(q), device=q.device)
    s_qd, s_qq = q @ docs.T, q @ q.T
    s_pos = s_qd[rows, pos_col][:, None]
    limit = s_pos.detach() + margin
    mask_qd = known_pos | (s_qd.detach() > limit)
    mask_qd[rows, pos_col] = False
    mask_qq = torch.eye(len(q), dtype=torch.bool, device=q.device) | (s_qq.detach() > limit)
    logits = torch.cat([s_qd.masked_fill(mask_qd, float("-inf")), s_qq.masked_fill(mask_qq, float("-inf"))], 1)
    n_neg = s_qd.numel() + s_qq.numel() - 2 * len(q)  # minus the positives and the diagonal
    masked = (mask_qd.sum() + mask_qq.sum() - len(q)).float() / n_neg
    return F.cross_entropy(logits / tau, pos_col), masked


def mine_negatives(teacher_q, docs, pos, neg, n_neg, chunk=256):
    """[N, n_neg] corpus rows per query: the stored negative first, then the n_neg - 1 corpus docs the teacher scores
    highest for that query (teacher query . teacher doc), never a labelled positive or the stored negative.
    n_neg = 1 is just the stored negatives (no scoring)."""
    out = np.array(neg)[:, None]
    if n_neg == 1:
        return out
    mined = []
    for start in range(0, len(neg), chunk):
        scores = teacher_q[start:start + chunk].to(docs.device) @ docs.T
        for r in range(len(scores)):
            scores[r, pos[start + r]] = float("-inf")
            scores[r, neg[start + r]] = float("-inf")
        mined.append(scores.topk(n_neg - 1, dim=1).indices.cpu().numpy())
    return np.concatenate([out, np.concatenate(mined)], 1)


def refine_main(rank, world_size, cfg):
    """One worker of ddp.launch(refine_main, cfg). Resumes from the latest step checkpoint; no-op if final/ exists."""
    cfg = resolve_cfg(REFINE_DEFAULTS, cfg)
    device = setup(rank, world_size)
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
    queries = data["train"]
    docs, ids = load_emb(cfg["teacher_emb_dir"], "corpus")
    assert ids == data["corpus"]["ids"], "corpus: teacher vectors don't match data.json"
    align_t, align_ids = load_emb(cfg["teacher_emb_dir"], "align")
    row = {qid: i for i, qid in enumerate(align_ids)}
    missing = [q for q in queries["ids"] if q not in row]
    assert not missing, f"{len(missing)} train queries have no teacher vector in align.npy"
    teacher_q = torch.from_numpy(align_t[[row[q] for q in queries["ids"]]].astype(np.float32))

    model, tokenizer = load_rank0_first(cfg["init_adapter"], device)  # frozen; the adapter is unfrozen below
    adapter_cfg = load_json(os.path.join(cfg["init_adapter"], ADAPTER_CONFIG))
    assert adapter_cfg["out_dim"] == cfg["dim"] == docs.shape[1], \
        f"adapter dim {adapter_cfg['out_dim']}, cfg dim {cfg['dim']}, teacher dim {docs.shape[1]}"
    if adapter_cfg.get("full"):  # a pruned, fully trained student: keep training every weight
        model.requires_grad_(True)
    else:
        lora, head = adapter_params(model)
        for p in lora + head:
            p.requires_grad_(True)
    docs = torch.from_numpy(docs).to(device)
    neg = mine_negatives(teacher_q, docs.float(), queries["pos"], queries["neg"], cfg["n_neg"])  # same on every rank
    batch_loss = functools.partial(_refine_batch_loss, tokenizer=tokenizer, queries=queries, neg=neg,
                                   docs=docs, teacher_q=teacher_q, cfg=cfg, device=device)
    run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, len(queries["ids"]), batch_loss, data, "refine")


def _refine_batch_loss(model, global_idx, local, epoch, tokenizer, queries, neg, docs, teacher_q, cfg, device):
    # docs of the whole global batch: positives (one per query, rotating over its positives by epoch), then negatives
    pos = [queries["pos"][i][epoch % len(queries["pos"][i])] for i in global_idx]
    cols = np.concatenate([pos, neg[global_idx].ravel()])
    idx = global_idx[local]
    known_pos = torch.from_numpy(np.stack([np.isin(cols, queries["pos"][i]) for i in idx])).to(device)
    pos_col = torch.arange(len(global_idx), device=device)[local]
    known_pos[torch.arange(len(idx), device=device), pos_col] = False

    enc = tokenize(tokenizer, [queries["texts"][i] for i in idx], cfg["max_query_len"], cfg["query_prefix"])
    q = embed(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), cfg["dim"]).float()
    cr, masked = contrastive_loss(q, docs[torch.from_numpy(cols).to(device)].float(), pos_col, known_pos,
                                  cfg["tau"], cfg["margin"])
    l2 = l2_loss(q, teacher_q[idx].to(device))
    return {"loss": cr + cfg["l2_weight"] * l2, "cr": cr, "l2": l2, "masked": masked}

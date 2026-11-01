"""Contrastive refinement: continue training the phase-2 adapter (LoRA + head; base frozen) with the Qwen3-Embedding
loss (arXiv 2506.05176, Eq. 1) over the global batch of G queries,
L_CR = -1/B sum_i log exp(s(q_i, d_i+)/tau) / Z_i,
Z_i = exp(s(q_i, d_i+)/tau) + sum_k m exp(s(q_i, d_ik-)/tau) + sum_{j!=i} m exp(s(q_i, q_j)/tau)
      + sum_{j!=i} m exp(s(d_i+, d_j+)/tau) + sum_{j!=i} m exp(s(q_i, d_j+)/tau),
s = dot of unit vectors, q = student queries (all-gathered with grad, see gather_queries), d = the frozen teacher
corpus vectors (corpus.npy, never re-encoded), plus l2_weight * L2(student q, teacher q). d_ik- = q_i's own n_neg
negatives: the stored negative[0] plus n_neg - 1 mined by teacher score (mine_negatives). m = 0 (false negative) for
q_i's labelled positives (so also a d_j+ equal to d_i+), q_i itself, and anything scoring above s(q_i, d_i+) + margin.
s(d_i+, d_j+) is teacher-teacher: no gradient, a per-query constant in Z_i.
Same loop, checkpoints and resume as align.py (align.run)."""
import functools
import os
import traceback

import numpy as np
import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

from align import final_exists, l2_loss, load_rank0_first, run
from data import load_emb, load_json, resolve_cfg
from ddp import cleanup, is_dist, setup
from model import ADAPTER_CONFIG, adapter_params, embed, gpu_mem, tokenize

REFINE_DEFAULTS = dict(
    init_adapter="runs/align/final",    # phase-2 adapter dir (LoRA + head trained further) or a pruned one (all weights)
    dim=256,                            # must equal the adapter's head output and the teacher index dim
    query_prefix="",                    # must match phase 2
    max_query_len=32,
    tau=0.05,                           # temperature
    margin=0.1,                         # negative with s > s(q, d+) + margin is masked as a false negative
    n_neg=1,                            # negatives per query: 1 = the stored one, k > 1 adds k-1 teacher-mined ones
    l2_weight=0.0,                      # weight of L2(student q, teacher q); 0 = pure L_CR (L2 is still logged)
    lr=5e-5,                            # LoRA params
    head_lr=1e-4,                       # head params
    weight_decay=0.0,
    warmup_ratio=0.05,                  # linear warmup, then cosine decay to 0
    epochs=1,
    batch_size=64,                      # queries per GPU; docs per step = 2 * batch_size * world_size
    grad_accum=1,                       # micro-batches per optimizer step; in-batch negatives stay per micro-batch
    grad_clip=1.0,
    amp=True,                           # fp16 autocast + GradScaler on CUDA
    seed=0,
    log_every=20,
    eval_every=200,                     # val L2/cos + val retrieval on rank 0, 0 = only at the end
    patience=0,                         # evals without a stop_on improvement before stopping, best weights -> final; 0 = off
    stop_on="val_ndcg@10",              # not val_loss: that is L2, which rises while CR improves nDCG
    save_every=500,                     # 0 = no resumable step checkpoints
    keep_last=2,
    data_path="cache/data.json",
    teacher_emb_dir="cache/emb/teacher",
    out_dir="runs/refine",
)


BLOCKS = ("qd", "neg", "qq", "dd")  # column blocks of the logits, in order


def contrastive_loss(all_q, rows, pos_docs, neg_docs, known_pos, known_neg, tau, margin):
    """L_CR (Qwen3 Eq. 1) for this rank's B queries all_q[rows] of a global batch of G (all unit norm).
    all_q [G, d] student queries (grad), rows [B] global index of this rank's queries, pos_docs [G, d] the teacher
    positive of each query, neg_docs [B, K, d] each local query's own negatives. known_pos [B, G] / known_neg [B, K]
    bool: that doc is a labelled positive of the query (covers duplicate positives). Logit blocks per query: q-d
    (its positive at column rows[i], the other queries' positives), own negatives, q-q, d-d (teacher-teacher, no grad).
    Masked: known positives, the query itself, and anything scoring above s(q_i, d_i+) + margin (detached).
    Returns (loss, stats): detached diagnostics, see contrastive_stats."""
    q = all_q[rows]
    s = dict(qd=q @ pos_docs.T, neg=torch.einsum("bd,bkd->bk", q, neg_docs), qq=q @ all_q.T,
             dd=pos_docs[rows] @ pos_docs.T)
    s_pos = s["qd"][torch.arange(len(q), device=q.device), rows]
    limit = s_pos.detach()[:, None] + margin
    self_ = F.one_hot(rows, len(all_q)).bool()
    mask = {k: v.detach() > limit for k, v in s.items()}
    mask["qd"] = (mask["qd"] | known_pos) & ~self_
    mask["neg"] = mask["neg"] | known_neg
    mask["qq"] = mask["qq"] | known_pos | self_
    mask["dd"] = mask["dd"] | known_pos | self_
    logits = torch.cat([s[k].masked_fill(mask[k], float("-inf")) for k in BLOCKS], 1)
    loss = F.cross_entropy(logits / tau, rows)
    return loss, contrastive_stats(s, mask, s_pos, self_, logits, rows, tau)


@torch.no_grad()
def contrastive_stats(s, mask, s_pos, self_, logits, rows, tau):
    """acc: share of queries whose positive is the top logit. s_pos: mean s(q_i, d_i+). gap: mean s_pos - hardest
    unmasked negative (rows with every negative masked skipped). m_<block>: masked share of that block's negatives
    (self not counted). z_<block>: that block's share of the negative mass of Z_i, mean over queries (sum to 1)."""
    stats = dict(acc=(logits.argmax(1) == rows).float().mean(), s_pos=s_pos.float().mean())
    neg_mask = {**mask, "qd": mask["qd"] | self_}  # the positive is not a negative
    neg = torch.cat([s[k].float().masked_fill(neg_mask[k], float("-inf")) for k in BLOCKS], 1)
    gap = s_pos.float() - neg.max(1).values
    finite = torch.isfinite(gap)  # +inf: every negative of that query masked
    stats["gap"] = torch.where(finite, gap, 0.0).sum() / finite.sum().clamp(min=1)
    for k in BLOCKS:
        n_self = 0 if k == "neg" else len(rows)  # q-d: the positive (never masked), q-q / d-d: the query (always)
        masked = mask[k].sum() - (0 if k in ("neg", "qd") else n_self)
        stats[f"m_{k}"] = masked.float() / max(mask[k].numel() - n_self, 1)
    p = torch.softmax(neg / tau, 1).nan_to_num(0.0)  # a row with every negative masked: all zero
    start = 0
    for k in BLOCKS:
        width = s[k].shape[1]
        stats[f"z_{k}"] = p[:, start:start + width].sum(1).mean()
        start += width
    return stats


def gather_queries(q):
    """[B, d] per rank -> [world_size * B, d] in rank order, differentiable: backward sums each rank's gradient back
    to the query's own rank. DDP's mean over ranks then gives the gradient of the global mean loss."""
    return torch.cat(dist_nn.all_gather(q)) if is_dist() else q


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
    print(f"[rank{rank}] corpus on {device}{gpu_mem(device)}, mining negatives", flush=True)
    neg = mine_negatives(teacher_q, docs.float(), queries["pos"], queries["neg"], cfg["n_neg"])  # same on every rank
    batch_loss = functools.partial(_refine_batch_loss, tokenizer=tokenizer, queries=queries, neg=neg,
                                   docs=docs, teacher_q=teacher_q, cfg=cfg, device=device)
    run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, len(queries["ids"]), batch_loss, data, "refine")


def _refine_batch_loss(model, global_idx, local, epoch, tokenizer, queries, neg, docs, teacher_q, cfg, device):
    # one positive per query of the whole global batch (rotating over its positives by epoch), indexed locally: every
    # rank holds the corpus and the same global_idx, so this equals an all_gather of the positives
    pos = np.array([queries["pos"][i][epoch % len(queries["pos"][i])] for i in global_idx])
    idx = global_idx[local]
    rows = torch.arange(len(global_idx), device=device)[local]
    known_pos = torch.from_numpy(np.stack([np.isin(pos, queries["pos"][i]) for i in idx])).to(device)
    known_neg = torch.from_numpy(np.stack([np.isin(neg[i], queries["pos"][i]) for i in idx])).to(device)

    enc = tokenize(tokenizer, [queries["texts"][i] for i in idx], cfg["max_query_len"], cfg["query_prefix"])
    q = embed(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), cfg["dim"]).float()
    cr, stats = contrastive_loss(gather_queries(q), rows, docs[torch.from_numpy(pos).to(device)].float(),
                                 docs[torch.from_numpy(neg[idx]).to(device)].float(), known_pos, known_neg,
                                 cfg["tau"], cfg["margin"])
    l2 = l2_loss(q, teacher_q[idx].to(device))
    return {"loss": cr + cfg["l2_weight"] * l2, "cr": cr, "l2": l2, **stats}

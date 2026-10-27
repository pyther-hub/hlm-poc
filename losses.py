"""Training losses, looked up by name: LOSSES[cfg["loss"]](q, batch, **cfg["loss_kwargs"]).

q is the student's unit-norm query embeddings [B, d] (fp32). batch is the prepared batch dict:
  target   [B, d]  teacher query embedding (unit norm)                     - align and CR batches
  pos_emb  [B, d]  teacher embedding of the query's positive doc, pos [B]  - CR batches only
  neg_emb  [B, d]  teacher embedding of its hard negative, neg [B]         - CR batches only
  extra_emb [E, d], extra [E]  random corpus docs                          - CR, when extra_negs > 0

To add a loss: write `def my_loss(q, batch, **kwargs) -> scalar tensor`, add it to LOSSES,
then set loss="my_loss", loss_kwargs={...} in the notebook's control panel.
"""
import torch
import torch.nn.functional as F


def l2(q, batch):
    """Mean squared L2 distance to the teacher query vector (both unit norm, so = 2 - 2cos)."""
    return ((q - batch["target"]) ** 2).sum(-1).mean()


def cosine(q, batch):
    return (1 - (q * batch["target"]).sum(-1)).mean()


def contrastive(q, batch, tau=0.05, n_hard_neg=1, same_tower=True, mask_margin=0.1, extra_negs=0):
    """Qwen3-Embedding-style InfoNCE against frozen teacher doc vectors.
    Candidates for query i: its positive, the other in-batch positives, the hard negatives (n_hard_neg is 0 or 1:
    the corpus keeps one negative per query), the other queries (same_tower), and extra_negs random corpus docs
    (sampled in train.prepare_cr). A candidate is masked out if it is query i's positive document or scores more
    than mask_margin above the positive (likely an unlabeled positive). The paper's doc-doc term is left out:
    the teacher docs are frozen, so it would give no gradient. Under DDP, negatives are per-rank (no all_gather)."""
    n = q.shape[0]
    docs, doc_rows = [batch["pos_emb"]], [batch["pos"]]
    if n_hard_neg:
        docs.append(batch["neg_emb"]), doc_rows.append(batch["neg"])
    if extra_negs:
        docs.append(batch["extra_emb"]), doc_rows.append(batch["extra"])
    docs, doc_rows = torch.cat(docs), torch.cat(doc_rows)

    s_pos = (q * batch["pos_emb"]).sum(-1, keepdim=True)
    own = torch.zeros(n, len(docs), dtype=torch.bool, device=q.device)
    own[:, :n] = torch.eye(n, dtype=torch.bool, device=q.device)  # column i is query i's positive
    s_qd = q @ docs.T
    false_neg = (doc_rows[None, :] == batch["pos"][:, None]) | (s_qd > s_pos + mask_margin)
    scores = [s_qd.masked_fill(false_neg & ~own, float("-inf"))]
    if same_tower:
        s_qq = q @ q.T
        eye = torch.eye(n, dtype=torch.bool, device=q.device)
        scores.append(s_qq.masked_fill(eye | (s_qq > s_pos + mask_margin), float("-inf")))
    logits = torch.cat(scores, dim=1) / tau
    return F.cross_entropy(logits, torch.arange(n, device=q.device))


def hybrid(q, batch, terms):
    """Weighted sum of named losses, e.g. terms={"l2": {"weight": 1.0}, "contrastive": {"weight": 0.1, "tau": 0.05}}."""
    total = 0.0
    for name, kw in terms.items():
        kw = dict(kw)
        total = total + kw.pop("weight", 1.0) * LOSSES[name](q, batch, **kw)
    return total


LOSSES = {"l2": l2, "cosine": cosine, "contrastive": contrastive, "hybrid": hybrid}

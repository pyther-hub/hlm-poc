"""Retrieval evaluation against the frozen teacher corpus index. Single process, one GPU.
Results are rows: name -> {split: metrics}, merged into <out_dir>/metrics.json."""
import math
import os
import time

import numpy as np
import torch

from data import load_emb, load_json, save_json
from model import DTYPES, encode, load_encoder, truncate_normalize
from teacher import _sync, free_gpu_memory


# ---------- sanity checks ----------

def check_emb(emb, ids, want_ids, name, tol=1e-2):
    """Cached embeddings are finite, unit norm and in data.json order."""
    assert np.isfinite(emb).all(), f"{name}: non-finite values"
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.abs(norms - 1).max() < tol, f"{name}: norms in [{norms.min():.4f}, {norms.max():.4f}], expected 1"
    assert ids == want_ids, f"{name}: ids don't match data.json"


def check_qrels(qrels, n_docs, name):
    """Every query has at least one positive and every positive is a valid corpus row."""
    bad = [i for i, pos in enumerate(qrels) if not pos or not all(0 <= p < n_docs for p in pos)]
    assert not bad, f"{name}: {len(bad)} queries with missing or out-of-range positives (first: {bad[0]})"


def load_index(data, emb_dir, splits):
    """Teacher corpus vectors + teacher query vectors per split, all sanity-checked. Returns (docs, {split: q})."""
    docs, ids = load_emb(emb_dir, "corpus")
    check_emb(docs, ids, data["corpus"]["ids"], "corpus")
    teacher_q = {}
    for split in splits:
        q, ids = load_emb(emb_dir, split)
        check_emb(q, ids, data[split]["ids"], split)
        check_qrels(data[split]["pos"], len(docs), split)
        teacher_q[split] = q
    print(f"[eval] index ok: corpus {docs.shape}, " + ", ".join(f"{s} {q.shape}" for s, q in teacher_q.items()))
    return docs, teacher_q


# ---------- retrieval + metrics ----------

@torch.no_grad()
def retrieve_topk(queries, docs, k=100, device=torch.device("cpu"), chunk=1024):
    """Brute-force top-k by dot product (unit-norm vectors -> cosine). fp16 matmul on GPU, fp32 on CPU.
    Returns (scores, idx) numpy arrays [N, k]."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    docs = torch.as_tensor(docs).to(device, dtype)
    k = min(k, len(docs))
    scores, idx = [], []
    for start in range(0, len(queries), chunk):
        q = torch.as_tensor(queries[start:start + chunk]).to(device, dtype)
        s, i = (q @ docs.T).float().topk(k, dim=1)
        scores.append(s.cpu()), idx.append(i.cpu())
    return torch.cat(scores).numpy(), torch.cat(idx).numpy()


def recall_at(k):
    return lambda ranked, pos: len(set(ranked[:k]) & pos) / len(pos)


def mrr_at(k):
    return lambda ranked, pos: next((1 / (r + 1) for r, d in enumerate(ranked[:k]) if d in pos), 0.0)


def ndcg_at(k):
    """Binary gains; ideal DCG puts all positives (up to k) first."""
    def f(ranked, pos):
        dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(ranked[:k]) if d in pos)
        return dcg / sum(1 / math.log2(r + 2) for r in range(min(len(pos), k)))
    return f


RECALL_KS = (1, 5, 10, 20, 50, 100)
METRICS = {**{f"recall@{k}": recall_at(k) for k in RECALL_KS}, "mrr@10": mrr_at(10), "ndcg@10": ndcg_at(10)}


def compute_metrics(topk_idx, qrels):
    """topk_idx [N, k] corpus rows, qrels: list of positive corpus rows per query. Mean of each METRICS entry."""
    pairs = [(list(ranked), set(pos)) for ranked, pos in zip(topk_idx, qrels)]
    return {name: float(np.mean([fn(ranked, pos) for ranked, pos in pairs])) for name, fn in METRICS.items()}


def truncate(x, dim):
    """MRL: first dim features, renormalised. Valid on cached vectors since they are already MRL-truncated."""
    return truncate_normalize(torch.as_tensor(np.asarray(x, np.float32)), dim).numpy()


def score(q_emb, docs, qrels, k=100, device=torch.device("cpu")):
    _, idx = retrieve_topk(q_emb, docs, k, device)
    return {**compute_metrics(idx, qrels), "n_queries": len(qrels)}


# ---------- runs ----------

def teacher_ceiling(data, docs, teacher_q, k=100, device=torch.device("cpu"), dims=()):
    """Teacher queries vs teacher corpus: the best a student can hope for. Row "teacher", plus one row
    "teacher@d<dim>" per extra MRL dim (cached vectors truncated + renormalised)."""
    runs = [("teacher", docs, teacher_q)]
    runs += [(f"teacher@d{dim}", truncate(docs, dim), {s: truncate(q, dim) for s, q in teacher_q.items()}) for dim in dims]
    return {name: {s: score(q, d, data[s]["pos"], k, device) for s, q in qs.items()} for name, d, qs in runs}


def evaluate_model(model, data, docs, splits=("val", "test"), dim=256, max_query_len=32, query_prefix="", k=100,
                   device=torch.device("cpu"), dtype="fp16", batch_size=256):
    """Embed each split's queries with `model` (hub id or local dir), retrieve from the teacher corpus.
    Returns {split: metrics}."""
    m, tok = load_encoder(model, device, DTYPES[dtype])
    result = {}
    for split in splits:
        q = encode(m, tok, data[split]["texts"], device, dim, max_query_len, query_prefix, batch_size)
        result[split] = score(q.numpy(), docs, data[split]["pos"], k, device)
    del m
    free_gpu_memory()
    return result


def save_metrics(out_dir, rows):
    """Merge rows (name -> {split: metrics}) into <out_dir>/metrics.json."""
    path = os.path.join(out_dir, "metrics.json")
    metrics = load_json(path) if os.path.exists(path) else {}
    metrics.update(rows)
    save_json(metrics, path)
    print(f"[eval] {', '.join(rows)} -> {path}")


def summarize(out_dir, split="test", retention_metric="ndcg@10"):
    """One table from <out_dir>/metrics.json, teacher row first. retention = row / teacher on retention_metric."""
    metrics = load_json(os.path.join(out_dir, "metrics.json"))
    rows = [{"name": name, **result[split]} for name, result in metrics.items() if split in result]
    rows.sort(key=lambda r: r["name"] != "teacher")
    teacher = next((r for r in rows if r["name"] == "teacher"), None)
    base = teacher[retention_metric] if teacher else 0
    cols = list(METRICS) + ["retention"]
    print(f"split={split}  retention = {retention_metric} / teacher {retention_metric}")
    print(f"{'run':<22}" + "".join(f"{c:>11}" for c in cols))
    for r in rows:
        r["retention"] = r[retention_metric] / base if base else float("nan")
        print(f"{r['name']:<22}" + "".join(f"{r[c]:>11.4f}" for c in cols))
    return rows


@torch.no_grad()
def measure_latency(model, tokenizer, texts, device, dim, max_len, prefix="", batch_sizes=(1, 8, 32, 128), repeats=3):
    """Query-encoding latency: median over repeats of the time to embed all texts at each batch size."""
    encode(model, tokenizer, texts[:64], device, dim, max_len, prefix, batch_size=16)  # warm-up
    rows = []
    for bs in batch_sizes:
        times = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            encode(model, tokenizer, texts, device, dim, max_len, prefix, batch_size=bs)
            _sync(device)
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        rows.append({"batch_size": bs, "ms_per_query": 1000 * t / len(texts), "queries_per_s": len(texts) / t})
        print(f"[latency] bs={bs:<4} {rows[-1]['ms_per_query']:8.3f} ms/query  {rows[-1]['queries_per_s']:10.1f} q/s")
    return rows

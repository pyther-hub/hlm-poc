"""Retrieval evaluation against the frozen teacher corpus index. Single process."""
import math
import os
import time

import numpy as np
import torch

from data import load_emb, load_json, save_json
from model import encode, load_encoder

DEBUG_QUERIES = 64


@torch.no_grad()
def retrieve_topk(queries, docs, k=100, device=torch.device("cpu"), chunk=1024):
    """Brute-force top-k by dot product (unit-norm vectors → cosine). fp16 matmul on GPU, fp32 on CPU.
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


METRICS = {"recall@100": recall_at(100), "mrr@10": mrr_at(10), "ndcg@10": ndcg_at(10)}


def compute_metrics(topk_idx, qrels):
    """topk_idx [N, k] corpus rows, qrels: list of positive corpus rows per query. Mean of each METRICS entry."""
    out = {}
    for name, fn in METRICS.items():
        out[name] = float(np.mean([fn(list(ranked), set(pos)) for ranked, pos in zip(topk_idx, qrels)]))
    return out


def _split_metrics(q_emb, teacher_q, docs, qrels, k, device):
    _, idx = retrieve_topk(q_emb, docs, k, device)
    m = compute_metrics(idx, qrels)
    m["cos_to_teacher"] = float((np.asarray(q_emb, np.float32) * np.asarray(teacher_q, np.float32)).sum(-1).mean())
    m["n_queries"] = len(qrels)
    return m


def _write_metrics(out_dir, name, result):
    path = os.path.join(out_dir, "metrics.json")
    metrics = load_json(path) if os.path.exists(path) else {}
    metrics[name] = result
    save_json(metrics, path)
    print(f"[eval] {name} -> {path}")


def teacher_ceiling(data_path, emb_dir, out_dir, splits=("val", "test"), k=100, device=torch.device("cpu"), debug=False):
    """Teacher queries vs teacher corpus: the best the student can hope for."""
    data, (docs, _) = load_json(data_path), load_emb(emb_dir, "corpus")
    result = {}
    for split in splits:
        q, _ = load_emb(emb_dir, split)
        n = DEBUG_QUERIES if debug else len(q)
        result[split] = _split_metrics(q[:n], q[:n], docs, data[split]["pos"][:n], k, device)
    _write_metrics(out_dir, "teacher", result)
    return result


def evaluate_dir(path, data_path, emb_dir, dim, max_query_len, device, out_dir=None, name=None,
                 splits=("val", "test"), k=100, batch_size=256, doc_emb_path=None, debug=False):
    """Embed val/test queries with the student at `path` (checkpoint dir, hub id or "tiny:<seed>"), retrieve top-k
    from the cached teacher corpus (or doc_emb_path, e.g. the student's native-space corpus) and write
    metrics.json into out_dir (default: path). Returns {split: metrics}."""
    out_dir = out_dir or path
    name = name or os.path.basename(os.path.normpath(path))
    data = load_json(data_path)
    docs = np.load(doc_emb_path) if doc_emb_path else load_emb(emb_dir, "corpus")[0]
    model, tokenizer = load_encoder(path, device)
    result = {}
    for split in splits:
        teacher_q, _ = load_emb(emb_dir, split)
        n = DEBUG_QUERIES if debug else len(teacher_q)
        q = encode(model, tokenizer, data[split]["texts"][:n], device, dim, max_query_len, batch_size=batch_size)
        result[split] = _split_metrics(q.numpy(), teacher_q[:n], docs, data[split]["pos"][:n], k, device)
    del model
    _write_metrics(out_dir, name, result)
    return result


def summarize_runs(dirs, split="test", retention_metric="ndcg@10"):
    """One table from the metrics.json in each dir, teacher row first. Returns the rows."""
    rows = []
    for d in dirs:
        path = os.path.join(d, "metrics.json")
        if not os.path.exists(path):
            print(f"[summary] no metrics.json in {d}, skipped")
            continue
        for name, result in load_json(path).items():
            if split in result:
                rows.append({"name": name, **result[split]})
    rows.sort(key=lambda r: r["name"] != "teacher")
    teacher = next((r for r in rows if r["name"] == "teacher"), None)
    cols = ["recall@100", "mrr@10", "ndcg@10", "retention", "cos_to_teacher"]
    print(f"split={split}  retention = {retention_metric} / teacher {retention_metric}")
    print(f"{'run':<22}" + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        r["retention"] = r[retention_metric] / teacher[retention_metric] if teacher and teacher[retention_metric] else float("nan")
        print(f"{r['name']:<22}" + "".join(f"{r[c]:>16.4f}" for c in cols))
    return rows


@torch.no_grad()
def measure_latency(model, tokenizer, texts, device, dim, max_len, batch_sizes=(1, 4, 8, 16, 32, 64), repeats=3):
    """Query-encoding latency: median over repeats of the time to embed all texts at each batch size."""
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    encode(model, tokenizer, texts[:64], device, dim, max_len, batch_size=16)  # warm-up
    rows = []
    for bs in batch_sizes:
        times = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            encode(model, tokenizer, texts, device, dim, max_len, batch_size=bs)
            sync()
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        rows.append({"batch_size": bs, "ms_per_query": 1000 * t / len(texts), "queries_per_s": len(texts) / t})
        print(f"bs={bs:<3} {rows[-1]['ms_per_query']:.3f} ms/query  {rows[-1]['queries_per_s']:.0f} q/s")
    return rows

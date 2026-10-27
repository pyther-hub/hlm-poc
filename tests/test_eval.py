"""Evaluation: retrieval, metrics by hand, sanity checks, and the notebook flow on a tiny cached index.
The index is built in-process with encode (no DDP) as one shard each, merged by teacher.merge_shards."""
import math
import os

import numpy as np
import pytest
import torch

from conftest import tiny_rows
from data import build_data
from evaluation import (check_emb, compute_metrics, evaluate_model, load_index, measure_latency, retrieve_topk,
                        save_metrics, summarize, teacher_ceiling, truncate)
from model import encode, load_encoder
from teacher import QUERY_PROMPT, merge_shards, shard_path

DEVICE = torch.device("cpu")
DIM = 32


@pytest.fixture(scope="module")
def index(tmp_path_factory, tiny_model_dir):
    root = str(tmp_path_factory.mktemp("eval"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    emb_dir = os.path.join(root, "emb")
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    for name in ("corpus", "val", "test"):
        prefix = "" if name == "corpus" else QUERY_PROMPT
        emb = encode(model, tok, data[name]["texts"], DEVICE, DIM, 32, prefix)
        os.makedirs(emb_dir, exist_ok=True)
        np.save(shard_path(emb_dir, name, 0, 1), emb.numpy())
        merge_shards(emb_dir, name, 1, data[name]["ids"])
    return root, data, emb_dir


def test_retrieve_topk_matches_numpy():
    rng = np.random.default_rng(0)
    q, d = rng.standard_normal((7, 16)).astype(np.float32), rng.standard_normal((50, 16)).astype(np.float32)
    scores, idx = retrieve_topk(q, d, k=10, chunk=3)
    want = np.argsort(-(q @ d.T), axis=1)[:, :10]
    assert (idx == want).all() and np.allclose(scores, np.take_along_axis(q @ d.T, want, 1), atol=1e-5)


def test_compute_metrics_by_hand():
    # query 0: positives {1, 5}, 1 found at rank 2; query 1: positive 4 never retrieved
    m = compute_metrics(np.array([[3, 1, 2], [9, 8, 7]]), [[1, 5], [4]])
    assert m["recall@1"] == 0 and m["recall@5"] == pytest.approx(0.25) and m["mrr@10"] == pytest.approx(0.25)
    ndcg0 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["ndcg@10"] == pytest.approx(ndcg0 / 2)


def test_truncate_renormalizes():
    x = truncate(np.random.default_rng(0).standard_normal((4, 32)), 8)
    assert x.shape == (4, 8) and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)


def test_check_emb_rejects_bad_inputs():
    emb = truncate(np.random.default_rng(0).standard_normal((3, 8)), 8)
    check_emb(emb, ["a", "b", "c"], ["a", "b", "c"], "ok")
    with pytest.raises(AssertionError, match="norms"):
        check_emb(emb * 2, ["a", "b", "c"], ["a", "b", "c"], "x")
    with pytest.raises(AssertionError, match="ids"):
        check_emb(emb, ["a", "c", "b"], ["a", "b", "c"], "x")


def test_eval_flow(index, tiny_model_dir):
    """Notebook cells: load + check index, teacher ceiling + MRL sweep, a student, metrics.json, summary."""
    root, data, emb_dir = index
    docs, teacher_q = load_index(data, emb_dir, ("val", "test"))
    rows = teacher_ceiling(data, docs, teacher_q, dims=(8, 16))
    assert set(rows) == {"teacher", "teacher@d8", "teacher@d16"}
    # the teacher model itself, through evaluate_model, reproduces the cached ceiling
    same = evaluate_model(tiny_model_dir, data, docs, dim=DIM, max_query_len=32, query_prefix=QUERY_PROMPT, dtype="fp32")
    assert same["val"]["mrr@10"] == pytest.approx(rows["teacher"]["val"]["mrr@10"], abs=0.05)
    rows["student"] = evaluate_model(tiny_model_dir, data, docs, dim=DIM, max_query_len=16, dtype="fp32")
    out_dir = os.path.join(root, "runs")
    save_metrics(out_dir, rows)
    table = summarize(out_dir, split="test")
    assert table[0]["name"] == "teacher" and table[0]["retention"] == pytest.approx(1.0)
    assert {r["name"] for r in table} == {"teacher", "teacher@d8", "teacher@d16", "student"}
    assert all(0 <= r["recall@1"] <= r["recall@100"] <= 1 for r in table)


def test_measure_latency_runs(tiny_model_dir):
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    rows = measure_latency(model, tok, ["a short query"] * 8, DEVICE, DIM, 16, batch_sizes=(1, 4), repeats=1)
    assert [r["batch_size"] for r in rows] == [1, 4] and all(r["ms_per_query"] > 0 for r in rows)

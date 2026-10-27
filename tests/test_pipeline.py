"""Multi-process checks (2 CPU processes, gloo) and the end-to-end tiny run of the 00 -> 01 -> 02 notebook logic.
Workers are imported from the repo modules (mp.spawn can't pickle test-local functions)."""
import os

import numpy as np
import pytest
import torch

from data import build_data, load_emb, load_json, save_json, tiny_rows
from ddp import launch
from evaluation import evaluate_dir, summarize_runs, teacher_ceiling
from model import encode, load_encoder
from teacher import QUERY_PROMPT, embed_main
from train import align_main, benchmark_scaling, cr_main

DEVICE = torch.device("cpu")
DIM = 32


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """00 cells 1-2: build data, embed it with the (tiny) teacher on 2 processes."""
    root = str(tmp_path_factory.mktemp("pipe"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    teacher = dict(model="tiny:1", dim=DIM, dtype="fp32", batch_size=8, doc_max_len=32, query_max_len=32,
                   data_path=os.path.join(root, "data.json"), emb_dir=os.path.join(root, "emb"))
    launch(embed_main, teacher, nprocs=2)
    common = dict(init="tiny:0", dim=DIM, max_query_len=16, batch_size=4, amp=False, debug=True,
                  data_path=teacher["data_path"], emb_dir=teacher["emb_dir"])
    return root, data, teacher, common


def test_sharded_embedding_matches_single_process(env):
    _, data, teacher, _ = env
    model, tok = load_encoder("tiny:1", DEVICE)
    for name in ("corpus", "align", "val", "test"):
        emb, ids = load_emb(teacher["emb_dir"], name)
        prefix = "" if name == "corpus" else QUERY_PROMPT
        want = encode(model, tok, data[name]["texts"], DEVICE, DIM, 32, prefix, batch_size=5)
        assert ids == data[name]["ids"] and emb.dtype == np.float16
        assert np.allclose(emb.astype(np.float32), want.float().numpy(), atol=2e-3), name


def test_end_to_end_tiny(env):
    """00 baselines -> 01 align -> 02 cr_only + aligned_cr -> 00 summary. Debug runs also assert
    (inside run_stage) that DDP weights are identical across ranks after align_main and cr_main."""
    root, _, teacher, common = env
    runs = os.path.join(root, "runs")
    baselines = os.path.join(runs, "baselines")
    ev = dict(data_path=teacher["data_path"], emb_dir=teacher["emb_dir"], dim=DIM, max_query_len=16, device=DEVICE)

    teacher_ceiling(teacher["data_path"], teacher["emb_dir"], baselines)
    evaluate_dir("tiny:0", out_dir=baselines, name="untrained_student", **ev)

    launch(align_main, {**common, "out_dir": os.path.join(runs, "aligned")}, nprocs=2)
    launch(cr_main, {**common, "out_dir": os.path.join(runs, "cr_only"),
                     "loss_kwargs": dict(tau=0.05, n_hard_neg=1, same_tower=True, mask_margin=0.1, extra_negs=4)},
           nprocs=2)
    launch(cr_main, {**common, "init": os.path.join(runs, "aligned"), "out_dir": os.path.join(runs, "aligned_cr")},
           nprocs=2)
    for d in ("aligned", "cr_only", "aligned_cr"):
        path = os.path.join(runs, d)
        assert {"cfg.json", "train_log.jsonl", "config.json"} <= set(os.listdir(path))
        evaluate_dir(path, **ev)

    rows = summarize_runs([baselines] + [os.path.join(runs, d) for d in ("aligned", "cr_only", "aligned_cr")])
    assert [r["name"] for r in rows] == ["teacher", "untrained_student", "aligned", "cr_only", "aligned_cr"]
    assert rows[0]["recall@100"] == 1.0 or rows[0]["recall@100"] > 0.5  # tiny corpus: top-100 covers nearly all
    assert all(np.isfinite(r["ndcg@10"]) for r in rows)
    assert load_json(os.path.join(runs, "aligned", "cfg.json"))["dim"] == DIM


def test_cr_fails_fast_without_aligned_dir(env, tmp_path):
    _, _, _, common = env
    with pytest.raises(Exception, match="checkpoint dir not found"):
        launch(cr_main, {**common, "init": str(tmp_path / "aligned_missing"), "out_dir": str(tmp_path / "x")}, nprocs=1)


def test_benchmark_scaling_runs(env):
    _, _, _, common = env
    results = benchmark_scaling(align_main, {**common, "debug": False}, steps=2)
    assert set(results) == {1, 2} and all(v > 0 for v in results.values())

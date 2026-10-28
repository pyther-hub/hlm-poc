"""run_exp: the KL loss, LoRA optimizer groups unchanged by trainable_params, evaluate_runs resuming, and the plots."""
import os

import matplotlib
import pytest
import torch
import torch.nn.functional as F

import run_exp
from conftest import build_teacher_index, tiny_rows
from data import build_data
from evaluation import load_index
from model import adapter_params, attach_adapter, load_encoder, trainable_params

matplotlib.use("Agg")
DEVICE = torch.device("cpu")
DIM = 32


def test_kl_loss_zero_iff_same():
    torch.manual_seed(0)
    t = F.normalize(torch.randn(8, 16), dim=-1)
    assert run_exp.kl_loss(t, t).item() == pytest.approx(0, abs=1e-6)
    assert run_exp.kl_loss(F.normalize(torch.randn(8, 16), dim=-1), t).item() > 0.01


def test_lora_trainable_params_match_adapter_params(tiny_model_dir):
    """align.run's optimizer groups (trainable_params) are the old ones (adapter_params), same order: resume works."""
    model, _ = load_encoder(tiny_model_dir, DEVICE)
    attach_adapter(model, 4, 8, 0.0, ("q_proj", "down_proj"), DIM)
    body, head = trainable_params(model)
    lora, head_old = adapter_params(model)
    assert [id(p) for p in body] == [id(p) for p in lora] and [id(p) for p in head] == [id(p) for p in head_old]


def test_evaluate_runs_adds_teacher_and_resumes(tiny_model_dir, tmp_path, monkeypatch):
    data = build_data(tiny_rows(40), core_n=30, n_val=5, n_test=5, n_align=30, split_seed=1)
    build_teacher_index(str(tmp_path / "emb"), data, tiny_model_dir, DIM, ("corpus", "val", "test"))
    index = load_index(data, str(tmp_path / "emb"), ("val", "test"))
    ev = dict(splits=("val", "test"), dim=DIM, max_query_len=16, device=DEVICE, dtype="fp32")
    exp_dir = str(tmp_path / "exp")
    metrics = run_exp.evaluate_runs({"base": tiny_model_dir}, data, index, exp_dir, ev)
    assert set(metrics) == {"teacher", "base"} and metrics["teacher"]["test"]["recall@100"] == 1.0
    monkeypatch.setattr(run_exp, "evaluate_model", lambda *a, **k: pytest.fail("re-evaluated a finished row"))
    assert run_exp.evaluate_runs({"base": tiny_model_dir}, data, index, exp_dir, ev) == metrics

    run_exp.print_rows(metrics, ["base"])
    run_exp.save_fig(run_exp.plot_bars(metrics, ["teacher", "base"]), exp_dir, "bars")
    assert os.path.exists(os.path.join(exp_dir, "plots", "bars.png"))

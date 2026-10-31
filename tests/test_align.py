"""Alignment: the hook adapter, adapter-only checkpoints, and align_main on 2 CPU processes (train, keep_last,
resume from the latest step checkpoint)."""
import os
import shutil

import pytest
import torch

from align import align_main, l2_loss, read_log, step_checkpoints
from conftest import build_teacher_index, tiny_rows
from data import build_data, save_json
from ddp import launch
from evaluation import evaluate_model, load_index
from model import (ADAPTER_CONFIG, ADAPTER_WEIGHTS, adapter_params, attach_adapter, encode, load_encoder,
                   save_adapter, tokenize)

DEVICE = torch.device("cpu")
DIM = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_adapter_starts_as_base_plus_head(tiny_model_dir):
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    enc = tokenize(tok, ["river city", "gold tax moon"], 16)
    base = model(**enc).last_hidden_state
    attach_adapter(model, 4, 8, 0.0, TARGETS, DIM)
    assert torch.allclose(model(**enc).last_hidden_state, model.align_head(base), atol=1e-6)


def test_adapter_checkpoint_roundtrip(tiny_model_dir, tmp_path):
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    attach_adapter(model, 4, 8, 0.0, TARGETS, DIM)
    lora, head = adapter_params(model)
    assert len(lora) == 2 * 7 * 2 and len(head) == 2 and all(p.requires_grad for p in lora + head)
    torch.manual_seed(1)
    for p in lora:
        p.data.normal_(0, 0.1)  # B starts at zero; make the LoRA path matter
    path = str(tmp_path / "ckpt")
    save_adapter(model, path, dict(base=tiny_model_dir, rank=4, alpha=8, dropout=0.0, targets=list(TARGETS),
                                   out_dim=DIM))
    assert sorted(os.listdir(path)) == sorted([ADAPTER_CONFIG, ADAPTER_WEIGHTS])
    saved = torch.load(os.path.join(path, ADAPTER_WEIGHTS))
    assert len(saved) == len(lora) + len(head) and all(".lora_" in k or k.startswith("align_head.") for k in saved)
    texts = ["river city 1", "gold tax moon 2", "salt king 3"]
    want = encode(model.eval(), tok, texts, DEVICE, DIM, 16)
    loaded, tok2 = load_encoder(path, DEVICE)
    assert not any(p.requires_grad for p in loaded.parameters())
    assert torch.allclose(encode(loaded, tok2, texts, DEVICE, DIM, 16).float(), want.float(), atol=1e-3)


def test_l2_loss_is_two_minus_two_cos():
    s, t = (torch.nn.functional.normalize(torch.randn(5, 8), dim=-1) for _ in range(2))
    assert l2_loss(s, t).item() == pytest.approx((2 - 2 * (s * t).sum(-1)).mean().item(), abs=1e-6)


@pytest.fixture(scope="module")
def run(tmp_path_factory, tiny_model_dir):
    """Teacher index from the tiny model, then 2-process alignment of the same tiny model: 6 steps/epoch x 4."""
    root = str(tmp_path_factory.mktemp("align"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    emb_dir = os.path.join(root, "emb")
    build_teacher_index(emb_dir, data, tiny_model_dir, DIM, ("corpus", "align", "val", "test"))
    cfg = dict(model=tiny_model_dir, dim=DIM, lr=1e-2, head_lr=1e-2, warmup_ratio=0.0, epochs=4, batch_size=4,
               log_every=2, eval_every=12, save_every=5, keep_last=2, data_path=os.path.join(root, "data.json"),
               teacher_emb_dir=emb_dir, out_dir=os.path.join(root, "align"))
    launch(align_main, cfg, nprocs=2)
    return data, emb_dir, cfg


def test_align_trains_and_keeps_last_checkpoints(run):
    data, emb_dir, cfg = run
    log = read_log(cfg["out_dir"])
    train = [r["train_loss"] for r in log if "train_loss" in r]
    assert len(train) == 12 and sum(train[-3:]) < sum(train[:3])
    assert [r["step"] for r in log if "val_loss" in r] == [12, 24]
    assert [os.path.basename(p) for p in step_checkpoints(cfg["out_dir"])] == ["step_0000015", "step_0000020"]
    docs, _ = load_index(data, emb_dir, ("val",))
    result = evaluate_model(os.path.join(cfg["out_dir"], "final"), data, docs, ("val",), DIM, dtype="fp32")
    assert 0 <= result["val"]["ndcg@10"] <= 1


def test_align_resumes_from_latest_checkpoint(run):
    _, _, cfg = run
    shutil.rmtree(os.path.join(cfg["out_dir"], "final"))
    n_before = len(read_log(cfg["out_dir"]))
    launch(align_main, cfg, nprocs=2)
    resumed = [r["step"] for r in read_log(cfg["out_dir"])[n_before:] if "train_loss" in r]
    assert resumed == [22, 24] and os.path.isdir(os.path.join(cfg["out_dir"], "final"))


def test_grad_accum_matches_bigger_batch(run):
    _, _, cfg = run
    weights = []
    for bs, accum in ((4, 1), (2, 2)):
        out_dir = cfg["out_dir"] + f"_bs{bs}_accum{accum}"
        launch(align_main, {**cfg, "lora_dropout": 0.0, "lr": 1e-3, "head_lr": 1e-3, "epochs": 1, "batch_size": bs,
                            "grad_accum": accum, "eval_every": 0, "save_every": 0, "out_dir": out_dir}, nprocs=2)
        assert [r["step"] for r in read_log(out_dir) if "train_loss" in r] == [2, 4, 6]  # 50 // 8 steps
        weights.append(torch.load(os.path.join(out_dir, "final", ADAPTER_WEIGHTS)))
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.allclose(weights[0][k], weights[1][k], atol=1e-5) for k in weights[0])


def test_early_stop_keeps_best(run):
    data, emb_dir, cfg = run
    out_dir = cfg["out_dir"] + "_early_stop"
    launch(align_main, {**cfg, "lr": 0.5, "head_lr": 0.5, "eval_every": 1, "patience": 1, "save_every": 0,
                        "out_dir": out_dir}, nprocs=2)
    val = [r for r in read_log(out_dir) if "val_loss" in r]
    assert val[-1]["step"] < 24 and val[-1]["val_loss"] >= min(r["val_loss"] for r in val)  # stopped early
    assert sorted(os.listdir(out_dir)) == ["final", "log.jsonl"]  # best/ renamed to final/, no step checkpoints
    _, teacher_q = load_index(data, emb_dir, ("val",))
    model, tok = load_encoder(os.path.join(out_dir, "final"), DEVICE)
    q = encode(model, tok, data["val"]["texts"], DEVICE, DIM, 32).float()  # ALIGN_DEFAULTS max_query_len
    best = l2_loss(q, torch.from_numpy(teacher_q["val"]).float()).item()
    assert best == pytest.approx(min(r["val_loss"] for r in val), abs=1e-3)

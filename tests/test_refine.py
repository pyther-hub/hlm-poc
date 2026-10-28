"""Contrastive refinement: L_CR and its false-negative masking by hand, and refine_main on 2 CPU processes starting
from a saved adapter."""
import math
import os

import pytest
import torch
import torch.nn.functional as F

from align import read_log
from conftest import build_teacher_index, tiny_rows
from data import build_data, save_json
from ddp import launch
from evaluation import evaluate_model, load_index
from model import ADAPTER_WEIGHTS, attach_adapter, load_encoder, save_adapter
from refine import contrastive_loss, mine_negatives, refine_main

DEVICE = torch.device("cpu")
DIM = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_contrastive_loss_by_hand():
    torch.manual_seed(0)
    q, docs = F.normalize(torch.randn(2, 8), dim=-1), F.normalize(torch.randn(4, 8), dim=-1)
    pos_col, tau = torch.tensor([0, 1]), 0.1
    known_pos = torch.zeros(2, 4, dtype=torch.bool)
    known_pos[0, 3] = True  # doc 3 is another labelled positive of query 0
    loss, masked = contrastive_loss(q, docs, pos_col, known_pos, tau, margin=10.0)  # margin too big to mask
    want = 0.0
    for i in range(2):
        negs = [(q[i] @ docs[j]).item() for j in range(4) if j != pos_col[i] and not known_pos[i, j]]
        negs.append((q[i] @ q[1 - i]).item())
        pos = (q[i] @ docs[pos_col[i]]).item()
        want -= pos / tau - math.log(sum(math.exp(s / tau) for s in [pos] + negs))
    assert loss.item() == pytest.approx(want / 2, abs=1e-5)
    assert masked.item() == pytest.approx(1 / 8)  # 1 of 2 * (3 docs + 1 query) negatives


def test_contrastive_loss_masks_negatives_above_margin():
    torch.manual_seed(0)
    q, docs = F.normalize(torch.randn(1, 8), dim=-1), F.normalize(torch.randn(2, 8), dim=-1)
    pos_col = torch.tensor([0])
    near_dup = torch.cat([docs, q])  # extra doc identical to the query: scores 1 > s(q, d+) + 0.1
    assert (q @ docs[0]).item() + 0.1 < 1
    loss, masked = contrastive_loss(q, near_dup, pos_col, torch.zeros(1, 3, dtype=torch.bool), 0.05, margin=0.1)
    without, _ = contrastive_loss(q, docs, pos_col, torch.zeros(1, 2, dtype=torch.bool), 0.05, margin=0.1)
    assert loss.item() == pytest.approx(without.item(), abs=1e-6) and masked.item() == pytest.approx(1 / 2)


def test_mine_negatives():
    docs = torch.eye(5)
    teacher_q = torch.tensor([[0., 1, 2, 3, 4], [4., 3, 2, 1, 0]])  # q0 ranks docs 4,3,2,1,0; q1 ranks 0,1,2,3,4
    pos, neg = [[4], [0]], [1, 3]
    assert mine_negatives(teacher_q, docs, pos, neg, 1).tolist() == [[1], [3]]  # n_neg=1: only the stored negative
    got = mine_negatives(teacher_q, docs, pos, neg, 3, chunk=1)
    assert got.tolist() == [[1, 3, 2], [3, 1, 2]]  # stored first; mined skip positives and the stored negative


@pytest.fixture(scope="module")
def run(tmp_path_factory, tiny_model_dir):
    """Tiny teacher index, a phase-2 adapter saved straight from attach_adapter, then 2-process refinement."""
    root = str(tmp_path_factory.mktemp("refine"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    emb_dir = os.path.join(root, "emb")
    build_teacher_index(emb_dir, data, tiny_model_dir, DIM, ("corpus", "align", "val", "test"))
    model, _ = load_encoder(tiny_model_dir, DEVICE)
    torch.manual_seed(0)
    attach_adapter(model, 4, 8, 0.0, TARGETS, DIM)
    init = os.path.join(root, "align_final")
    save_adapter(model, init, dict(base=tiny_model_dir, rank=4, alpha=8, dropout=0.0, targets=list(TARGETS),
                                   out_dim=DIM))
    init_mtime = os.path.getmtime(os.path.join(init, ADAPTER_WEIGHTS))
    cfg = dict(init_adapter=init, dim=DIM, lr=1e-2, head_lr=1e-2, warmup_ratio=0.0, epochs=4, batch_size=4,
               log_every=3, eval_every=6, save_every=100, data_path=os.path.join(root, "data.json"),
               teacher_emb_dir=emb_dir, out_dir=os.path.join(root, "refine"))
    launch(refine_main, cfg, nprocs=2)
    return data, emb_dir, cfg, init_mtime


def test_refine_trains_from_adapter(run):
    data, emb_dir, cfg, init_mtime = run
    log = read_log(cfg["out_dir"])
    train = [r for r in log if "train_loss" in r]
    assert len(train) == 4 and all({"cr", "l2", "masked"} <= set(r) for r in train)  # 28 train queries / 8 = 3 x 4
    assert train[-1]["cr"] < train[0]["cr"]
    assert [r["step"] for r in log if "val_loss" in r] == [6, 12]
    assert os.path.getmtime(os.path.join(cfg["init_adapter"], ADAPTER_WEIGHTS)) == init_mtime  # init untouched
    docs, _ = load_index(data, emb_dir, ("val",))
    result = evaluate_model(os.path.join(cfg["out_dir"], "final"), data, docs, ("val",), DIM, dtype="fp32")
    assert 0 <= result["val"]["ndcg@10"] <= 1


def test_refine_with_mined_negatives(run):
    _, _, cfg, _ = run
    cfg = {**cfg, "n_neg": 3, "epochs": 1, "eval_every": 0, "out_dir": cfg["out_dir"] + "_mined"}
    launch(refine_main, cfg, nprocs=2)
    train = [r for r in read_log(cfg["out_dir"]) if "train_loss" in r]
    assert len(train) == 1 and all(map(math.isfinite, (train[0]["cr"], train[0]["l2"])))
    assert os.path.exists(os.path.join(cfg["out_dir"], "final"))

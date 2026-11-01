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
from ddp import cleanup, launch, setup
from evaluation import evaluate_model, load_index
from model import ADAPTER_WEIGHTS, attach_adapter, load_encoder, save_adapter
from refine import BLOCKS, contrastive_loss, gather_queries, mine_negatives, refine_main

DEVICE = torch.device("cpu")
DIM = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _batch(G=4, K=2, d=8, seed=0):
    """A global batch: all_q [G, d], pos_docs [G, d], neg_docs [G, K, d] (unit norm), no known positives."""
    g = torch.Generator().manual_seed(seed)
    unit = lambda *shape: F.normalize(torch.randn(*shape, generator=g), dim=-1)  # noqa: E731
    return unit(G, d), unit(G, d), unit(G, K, d), torch.zeros(G, G, dtype=torch.bool), torch.zeros(G, K, dtype=torch.bool)


def _eq1(all_q, rows, pos_docs, neg_docs, known_pos, known_neg, tau, margin):
    """Qwen3 Eq. 1 written out term by term, as python loops. Returns (loss, per-query {block: [unmasked scores]},
    per-query positive score)."""
    loss, kept, s_pos = 0.0, [], []
    for b, i in enumerate(rows.tolist()):
        q, pos = all_q[i], (all_q[i] @ pos_docs[i]).item()
        limit, blocks = pos + margin, {k: [] for k in BLOCKS}
        for j in range(len(all_q)):
            if j == i or known_pos[b, j]:
                continue
            for k, s in (("qd", q @ pos_docs[j]), ("qq", q @ all_q[j]), ("dd", pos_docs[i] @ pos_docs[j])):
                if s.item() <= limit:
                    blocks[k].append(s.item())
        for k in range(neg_docs.shape[1]):
            s = (q @ neg_docs[b, k]).item()
            if not known_neg[b, k] and s <= limit:
                blocks["neg"].append(s)
        negs = sum(blocks.values(), [])
        loss -= pos / tau - math.log(sum(math.exp(s / tau) for s in [pos] + negs))
        kept.append(blocks)
        s_pos.append(pos)
    return loss / len(rows), kept, s_pos


def test_contrastive_loss_by_hand():
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch()
    rows, tau = torch.tensor([2, 3]), 0.1  # this rank's queries, as rank 1 of 2 with B = 2
    known_pos[2, 0] = True  # query 0's positive is also a labelled positive of query 2
    known_neg[3, 1] = True  # a negative of query 3 is one of its labelled positives
    args = (all_q, rows, pos_docs, neg_docs[rows], known_pos[rows], known_neg[rows], tau)
    loss, stats = contrastive_loss(*args, margin=10.0)  # margin too big to mask
    want, kept, s_pos = _eq1(*args, margin=10.0)
    assert loss.item() == pytest.approx(want, abs=1e-5)
    assert [len(kept[0][k]) for k in BLOCKS] == [2, 2, 2, 2] and [len(kept[1][k]) for k in BLOCKS] == [3, 1, 3, 3]
    assert stats["s_pos"].item() == pytest.approx(sum(s_pos) / 2, abs=1e-6)
    hardest = [max(sum(blocks.values(), [])) for blocks in kept]
    assert stats["acc"].item() == sum(p > h for p, h in zip(s_pos, hardest)) / 2
    assert stats["gap"].item() == pytest.approx(sum(p - h for p, h in zip(s_pos, hardest)) / 2, abs=1e-5)
    # per block: masked / negatives over both queries (q-d, q-q, d-d: 3 per query, own negatives: 2)
    assert [stats[f"m_{k}"].item() for k in BLOCKS] == pytest.approx([1 / 6, 1 / 4, 1 / 6, 1 / 6])
    for k in BLOCKS:
        share = [sum(math.exp(s / tau) for s in blocks[k]) / sum(math.exp(s / tau) for s in sum(blocks.values(), []))
                 for blocks in kept]
        assert stats[f"z_{k}"].item() == pytest.approx(sum(share) / 2, abs=1e-5)
    assert sum(stats[f"z_{k}"].item() for k in BLOCKS) == pytest.approx(1.0, abs=1e-5)


def test_contrastive_loss_masks():
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch(G=3)
    rows = torch.tensor([0])
    q = all_q[0]
    all_q[2], pos_docs[1], neg_docs[0, 1] = q, q, q  # a near-duplicate of query 0 in q-q, q-d and its negatives
    assert (q @ pos_docs[0]).item() + 0.1 < 1  # each scores 1 > s(q, d+) + margin
    args = (all_q, rows, pos_docs, neg_docs[rows], known_pos[rows], known_neg[rows], 0.05)
    loss, stats = contrastive_loss(*args, margin=0.1)
    assert loss.item() == pytest.approx(_eq1(*args, margin=0.1)[0], abs=1e-5)
    assert stats["m_qd"].item() >= 0.5 and stats["m_qq"].item() >= 0.5 and stats["m_neg"].item() >= 0.5
    without, _ = contrastive_loss(all_q, rows, pos_docs, neg_docs[rows][:, :1], known_pos[rows],
                                  known_neg[rows][:, :1], 0.05, margin=0.1)
    assert loss.item() == pytest.approx(without.item(), abs=1e-6)  # a masked negative = an absent one

    # queries 0 and 1 share a positive: each masks the other's q-d, q-q and d-d entries
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch(G=2)
    pos_docs[1] = pos_docs[0]
    known_pos[:] = True  # caller: isin(positives of the batch, labelled positives of the query)
    _, stats = contrastive_loss(all_q, torch.tensor([0, 1]), pos_docs, neg_docs, known_pos, known_neg, 0.05, 10.0)
    assert [stats[f"m_{k}"].item() for k in ("qd", "qq", "dd")] == [1.0, 1.0, 1.0] and stats["m_neg"].item() == 0.0


def test_contrastive_loss_gradients():
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch()
    all_q.requires_grad_(True)
    rows = torch.tensor([2, 3])
    loss, stats = contrastive_loss(all_q, rows, pos_docs, neg_docs[rows], known_pos[rows], known_neg[rows], 0.05, 0.1)
    loss.backward()
    assert (all_q.grad.norm(dim=-1) > 0).all()  # rows 0, 1 belong to another rank: their gradient comes from q-q
    assert not pos_docs.requires_grad and not neg_docs.requires_grad  # teacher docs: plain tensors, frozen
    assert not any(v.requires_grad for v in stats.values())


def test_contrastive_loss_stable():
    v = F.normalize(torch.randn(8), dim=-1)
    G, K = 4, 2
    all_q, pos_docs, neg_docs = v.repeat(G, 1), v.repeat(G, 1), v.repeat(G, K, 1)
    known_pos, known_neg = torch.zeros(G, G, dtype=torch.bool), torch.zeros(G, K, dtype=torch.bool)
    rows = torch.arange(G)
    loss, _ = contrastive_loss(all_q, rows, pos_docs, neg_docs, known_pos, known_neg, 0.01, 0.1)  # all ties at 1/tau
    assert loss.item() == pytest.approx(math.log(1 + 3 * (G - 1) + K), abs=1e-4)
    loss, stats = contrastive_loss(all_q, rows, pos_docs, neg_docs, known_pos, known_neg, 0.01, -1.0)  # mask all
    assert loss.item() == pytest.approx(0.0, abs=1e-6) and stats["acc"].item() == 1.0
    assert all(math.isfinite(v.item()) for v in stats.values())
    assert [stats[f"m_{k}"].item() for k in BLOCKS] == [1.0] * 4


def _gather_worker(rank, world_size, cfg):
    """One rank of test_gather_matches_one_process: loss + query gradients of this rank's slice of _batch."""
    setup(rank, world_size)
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch(G=cfg["G"])
    all_q[4] = all_q[1]  # a duplicate query on the other rank: masked in q-q across ranks
    known_pos[0, 1] = True
    B = cfg["G"] // world_size
    rows = torch.arange(rank * B, (rank + 1) * B)
    q = all_q[rows].clone().requires_grad_(True)
    loss, _ = contrastive_loss(gather_queries(q), rows, pos_docs, neg_docs[rows], known_pos[rows], known_neg[rows],
                               0.05, 0.1)
    loss.backward()
    torch.save(dict(loss=loss.detach(), grad=q.grad), os.path.join(cfg["out"], f"rank{rank}.pt"))
    cleanup()


def test_gather_matches_one_process(tmp_path):
    G = 6
    launch(_gather_worker, dict(G=G, out=str(tmp_path)), nprocs=2)
    ranks = [torch.load(os.path.join(tmp_path, f"rank{r}.pt")) for r in range(2)]
    all_q, pos_docs, neg_docs, known_pos, known_neg = _batch(G=G)
    all_q[4] = all_q[1]
    known_pos[0, 1] = True
    all_q.requires_grad_(True)
    loss, _ = contrastive_loss(all_q, torch.arange(G), pos_docs, neg_docs, known_pos, known_neg, 0.05, 0.1)
    loss.backward()
    assert torch.stack([r["loss"] for r in ranks]).mean().item() == pytest.approx(loss.item(), abs=1e-6)
    # DDP averages parameter gradients over ranks: per-rank query gradients / world_size = the global mean's gradient
    assert torch.allclose(torch.cat([r["grad"] for r in ranks]) / 2, all_q.grad, atol=1e-6)


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
    # margin 10: no margin masks, so cr compares over the same negatives at every step. With the default 0.1 the tiny
    # random model starts with every q-q / d-d pair masked as a near-duplicate and cr rises as they unmask.
    cfg = dict(init_adapter=init, dim=DIM, margin=10.0, lr=1e-2, head_lr=1e-2, warmup_ratio=0.0, epochs=4,
               batch_size=4, log_every=3, eval_every=6, save_every=100, data_path=os.path.join(root, "data.json"),
               teacher_emb_dir=emb_dir, out_dir=os.path.join(root, "refine"))
    launch(refine_main, cfg, nprocs=2)
    return data, emb_dir, cfg, init_mtime


def test_refine_trains_from_adapter(run):
    data, emb_dir, cfg, init_mtime = run
    log = read_log(cfg["out_dir"])
    train = [r for r in log if "train_loss" in r]
    stat_keys = {"cr", "l2", "acc", "s_pos", "gap", "grad_norm"} | {f"{p}_{k}" for p in "mz" for k in BLOCKS}
    assert len(train) == 4 and all(stat_keys <= set(r) for r in train)  # 28 train queries / 8 = 3 x 4
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


def test_refine_early_stop_keeps_best_ndcg(run):
    data, emb_dir, cfg, _ = run
    cfg = {**cfg, "lr": 0.5, "head_lr": 0.5, "eval_every": 1, "patience": 1, "save_every": 0,
           "out_dir": cfg["out_dir"] + "_early_stop"}
    launch(refine_main, cfg, nprocs=2)
    val = [r for r in read_log(cfg["out_dir"]) if "val_ndcg@10" in r]
    assert val[-1]["step"] < 12 and val[-1]["val_ndcg@10"] <= max(r["val_ndcg@10"] for r in val)  # stopped early
    assert sorted(os.listdir(cfg["out_dir"])) == ["final", "log.jsonl"]  # best/ renamed to final/
    docs, _ = load_index(data, emb_dir, ("val",))
    result = evaluate_model(os.path.join(cfg["out_dir"], "final"), data, docs, ("val",), DIM, dtype="fp32")
    assert result["val"]["ndcg@10"] == pytest.approx(max(r["val_ndcg@10"] for r in val), abs=1e-4)

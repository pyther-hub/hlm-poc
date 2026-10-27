"""Fast in-process checks: model, data, losses, metrics, overfit-one-batch. CPU, offline, tiny random Qwen3."""
import math

import numpy as np
import pytest
import torch

from data import build_data, cr_triples, make_loader, resolve_cfg, tiny_rows
from evaluation import compute_metrics, retrieve_topk
from losses import LOSSES, contrastive
from model import embed, load_encoder, load_encoder_dir, save_encoder, tokenize, truncate_normalize
from train import ALIGN_DEFAULTS, CR_DEFAULTS, cosine_schedule, prepare_align, prepare_cr, train_loop

DEVICE = torch.device("cpu")
DIM = 32


@pytest.fixture(scope="module")
def student():
    return load_encoder("tiny:0", DEVICE, trainable="all_but_embeddings")


@pytest.fixture(scope="module")
def rows():
    return tiny_rows(60)


@pytest.fixture(scope="module")
def data(rows):
    return build_data(rows, core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)


# ---------- model ----------

def test_pooling_padding_invariant(student):
    model, tok = student
    alone = tokenize(tok, ["gold river"], 16)
    padded = tokenize(tok, ["gold river", "a much longer query about the moon and the salt lake"], 16)
    assert padded["attention_mask"][0, 0] == 0, "expected left padding"
    a = embed(model, alone["input_ids"], alone["attention_mask"], DIM)
    b = embed(model, padded["input_ids"], padded["attention_mask"], DIM)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def test_tokenize_ends_with_eot(student):
    _, tok = student
    enc = tokenize(tok, ["one two three four five six seven"], max_length=4)
    assert enc["input_ids"].shape[1] == 4 and enc["input_ids"][0, -1] == tok.pad_token_id


def test_truncate_normalize_unit_norm():
    x = truncate_normalize(torch.randn(5, 64) * 7, 16)
    assert x.shape == (5, 16) and torch.allclose(x.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_trainable_freezes_embeddings(student):
    model, _ = student
    assert not model.get_input_embeddings().weight.requires_grad
    assert all(p.requires_grad for n, p in model.named_parameters() if "embed_tokens" not in n)


def test_save_load_roundtrip(student, tmp_path):
    model, tok = student
    save_encoder(model, tok, str(tmp_path / "ckpt"), dtype="fp32")
    model2, tok2 = load_encoder_dir(str(tmp_path / "ckpt"), DEVICE)
    enc = tokenize(tok, ["salt lake city"], 16)
    assert torch.allclose(embed(model, **enc, dim=DIM), embed(model2, **tokenize(tok2, ["salt lake city"], 16), dim=DIM))
    with pytest.raises(FileNotFoundError):
        load_encoder_dir(str(tmp_path / "missing"), DEVICE)


# ---------- data ----------

def test_splits_disjoint_by_query(data):
    train, val, test = (set(data[s]["ids"]) for s in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert len(val) == 6 and len(test) == 6 and len(train) == 28


def test_corpus_has_positives_and_train_negatives(rows, data):
    corpus = data["corpus"]["ids"]
    assert len(corpus) == len(set(corpus)), "corpus must be deduplicated"
    by_id = {r["query_id"]: r for r in rows}
    for split in ("train", "val", "test"):
        for qid, pos, neg in zip(data[split]["ids"], data[split]["pos"], data[split]["neg"]):
            assert [corpus[p] for p in pos] == [p["docid"] for p in by_id[qid]["positives"]]
            assert corpus[neg] == by_id[qid]["negative"]["docid"]


def test_alignment_queries_exclude_val_test(data):
    align = set(data["align"]["ids"])
    assert set(data["train"]["ids"]) <= align
    assert not align & (set(data["val"]["ids"]) | set(data["test"]["ids"]))
    held_texts = set(data["val"]["texts"]) | set(data["test"]["texts"])
    assert not set(data["align"]["texts"]) & held_texts
    assert len(align) == 48  # 28 train + all 20 "other" queries (asked for 50)


def test_cr_triples(data):
    t = cr_triples(data)
    assert len(t) == len(data["train"]["ids"]) and all(isinstance(p, int) and isinstance(n, int) for _, p, n in t)


def test_resolve_cfg_rejects_typos():
    with pytest.raises(KeyError):
        resolve_cfg(ALIGN_DEFAULTS, {"learning_rate": 1e-3})
    assert resolve_cfg(CR_DEFAULTS, {"loss": "l2", "loss_kwargs": {}})["loss_kwargs"] == {}


# ---------- losses ----------

def _cr_batch(n, d, n_docs=20, extra=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    docs = torch.nn.functional.normalize(torch.randn(n_docs, d, generator=g), dim=-1)
    pos, neg = torch.randperm(n_docs, generator=g)[:n], torch.randperm(n_docs, generator=g)[:n]
    ex = torch.randint(n_docs, (extra,), generator=g)
    return {"target": torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1),
            "pos": pos, "neg": neg, "pos_emb": docs[pos], "neg_emb": docs[neg], "extra": ex, "extra_emb": docs[ex]}


@pytest.mark.parametrize("name", list(LOSSES))
def test_every_loss_finite_with_grads(name):
    batch = _cr_batch(6, 8)
    q = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1).requires_grad_()
    kw = {"hybrid": {"terms": {"l2": {"weight": 1.0}, "contrastive": {"weight": 0.5, "extra_negs": 3}}},
          "contrastive": {"extra_negs": 3}}.get(name, {})
    loss = LOSSES[name](q, batch, **kw)
    loss.backward()
    assert torch.isfinite(loss) and loss.ndim == 0 and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_l2_zero_on_equal_inputs():
    t = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    assert LOSSES["l2"](t.clone(), {"target": t}).item() == 0.0


def test_contrastive_masks_same_document():
    # q0 = pos(doc 0); the hard negative of q1 is doc 0 as well -> masked for q0. Same for q1 / doc 1.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch = {"pos": torch.tensor([0, 1]), "neg": torch.tensor([1, 0]),
             "pos_emb": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "neg_emb": torch.tensor([[0.0, 1.0], [1.0, 0.0]])}
    got = contrastive(q, batch, tau=1.0, n_hard_neg=1, same_tower=False, mask_margin=10.0)
    # row 0 candidates: doc0 (pos, s=1), doc1 (s=0), neg doc1 (s=0), neg doc0 (masked) -> -log(e / (e + 2))
    assert math.isclose(got.item(), math.log((math.e + 2) / math.e), rel_tol=1e-6)


def test_contrastive_masks_by_margin():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch = {"pos": torch.tensor([5, 7]), "neg": torch.tensor([5, 7]),
             "pos_emb": torch.tensor([[0.6, 0.8], [1.0, 0.0]]), "neg_emb": torch.tensor([[0.6, 0.8], [1.0, 0.0]])}
    kw = dict(tau=1.0, n_hard_neg=0, same_tower=False)
    # row 0: s_pos=0.6, other doc scores 1.0 > 0.6 + 0.1 -> masked; row 1: s_pos=0, other 0.8 > 0.1 -> masked
    assert contrastive(q, batch, mask_margin=0.1, **kw).item() == pytest.approx(0.0, abs=1e-7)
    loose = contrastive(q, batch, mask_margin=1.0, **kw).item()
    want = (-math.log(math.exp(0.6) / (math.exp(0.6) + math.e)) - math.log(1 / (math.exp(0.8) + 1))) / 2
    assert loose == pytest.approx(want, rel=1e-6)


# ---------- metrics ----------

def test_retrieve_topk_matches_numpy():
    rng = np.random.default_rng(0)
    q, d = rng.standard_normal((7, 16)).astype(np.float32), rng.standard_normal((50, 16)).astype(np.float32)
    scores, idx = retrieve_topk(q, d, k=10, chunk=3)
    want = np.argsort(-(q @ d.T), axis=1)[:, :10]
    assert (idx == want).all() and np.allclose(scores, np.take_along_axis(q @ d.T, want, 1), atol=1e-5)


def test_compute_metrics_by_hand():
    m = compute_metrics(np.array([[3, 1, 2], [9, 8, 7]]), [[1, 5], [4]])
    assert m["recall@100"] == pytest.approx((0.5 + 0.0) / 2)
    assert m["mrr@10"] == pytest.approx((0.5 + 0.0) / 2)
    ndcg0 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["ndcg@10"] == pytest.approx(ndcg0 / 2)


# ---------- overfit one batch (the real train_loop, no DDP) ----------

def _overfit(cfg_defaults, items, prepare, ctx, steps=40, lr=3e-3):
    model, tok = load_encoder("tiny:0", DEVICE, trainable="all_but_embeddings")
    loader = make_loader(items, tok, batch_size=len(items), max_len=16, shuffle=False)
    cfg = {**cfg_defaults, "dim": DIM, "epochs": steps, "val_every": 10**9, "amp": False}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = cosine_schedule(opt, steps, 0, 1.0)
    log = train_loop(model, loader, opt, sched, DEVICE, {**cfg, "val_every": 1}, lambda b, d: prepare(b, d, ctx, cfg))
    return log[0]["train_loss"], log[-1]["train_loss"]


def test_overfit_one_batch_align():
    rng = np.random.default_rng(0)
    targets = rng.standard_normal((8, DIM)).astype(np.float32)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    items = [{"text": f"query number {i} about gold", "target": targets[i]} for i in range(8)]
    first, last = _overfit(ALIGN_DEFAULTS, items, prepare_align, None)
    assert last < 0.3 * first, (first, last)


def test_overfit_one_batch_cr():
    docs = torch.nn.functional.normalize(torch.randn(30, DIM, generator=torch.Generator().manual_seed(0)), dim=-1)
    items = [{"text": f"query number {i} about rain", "pos": i, "neg": 10 + i, "target": docs[i].numpy()}
             for i in range(8)]
    ctx = {"docs": docs, "gen": torch.Generator().manual_seed(0)}
    first, last = _overfit(CR_DEFAULTS, items, prepare_cr, ctx)
    assert last < 0.3 * first, (first, last)

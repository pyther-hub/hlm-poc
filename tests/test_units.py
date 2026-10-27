"""Fast in-process checks: model, data, cfg. CPU, offline, tiny random Qwen3."""
import pytest
import torch

from conftest import tiny_rows
from data import build_data, resolve_cfg
from model import embed, encode, load_encoder, tokenize, truncate_normalize
from teacher import TEACHER_DEFAULTS

DEVICE = torch.device("cpu")
DIM = 32


@pytest.fixture(scope="module")
def encoder(tiny_model_dir):
    return load_encoder(tiny_model_dir, DEVICE)


@pytest.fixture(scope="module")
def rows():
    return tiny_rows(60)


@pytest.fixture(scope="module")
def data(rows):
    return build_data(rows, core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)


# ---------- model ----------

def test_pooling_padding_invariant(encoder):
    model, tok = encoder
    alone = tokenize(tok, ["gold river"], 16)
    padded = tokenize(tok, ["gold river", "a much longer query about the moon and the salt lake"], 16)
    assert padded["attention_mask"][0, 0] == 0, "expected left padding"
    a = embed(model, alone["input_ids"], alone["attention_mask"], DIM)
    b = embed(model, padded["input_ids"], padded["attention_mask"], DIM)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def test_tokenize_ends_with_eot(encoder):
    _, tok = encoder
    enc = tokenize(tok, ["one two three four five six seven"], max_length=4)
    assert enc["input_ids"].shape[1] == 4 and enc["input_ids"][0, -1] == tok.pad_token_id


def test_truncate_normalize_unit_norm():
    x = truncate_normalize(torch.randn(5, 64) * 7, 16)
    assert x.shape == (5, 16) and torch.allclose(x.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_encode_progress_prints_lines(encoder, capsys):
    model, tok = encoder
    encode(model, tok, [f"text {i}" for i in range(10)], DEVICE, DIM, 16, batch_size=2, progress=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[encode]")]
    assert len(lines) == 5 and lines[-1].startswith("[encode] 5/5 batches")


def test_encoder_is_frozen(encoder):
    model, _ = encoder
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_missing_model_dir_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_encoder(str(tmp_path / "missing"), DEVICE)


# ---------- data ----------

def test_splits_disjoint_by_query(data):
    train, val, test = (set(data[s]["ids"]) for s in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert len(val) == 6 and len(test) == 6 and len(train) == 28


def test_corpus_has_positives_and_negatives(rows, data):
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


def test_resolve_cfg_rejects_typos():
    with pytest.raises(KeyError):
        resolve_cfg(TEACHER_DEFAULTS, {"batchsize": 32})
    assert resolve_cfg(TEACHER_DEFAULTS, {"dim": 128})["dim"] == 128

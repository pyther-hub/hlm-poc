"""BERT-style students (bge / e5: BERT, DenseOn / granite r2: ModernBERT): right padding, the model's own pooling,
the head on the last LayerNorm / final_norm, and a full fine-tune (align full=True) on 2 CPU processes followed by
contrastive refinement of every weight."""
import os

import pytest
import torch
import torch.nn.functional as F

from align import align_main, read_log
from conftest import build_teacher_index, tiny_rows
from data import build_data, load_json, save_json
from ddp import launch
from evaluation import evaluate_model, load_index
from model import ADAPTER_CONFIG, ADAPTER_WEIGHTS, attach_head, embed, encode, load_encoder, tokenize
from refine import refine_main

DEVICE = torch.device("cpu")
DIM = 32


def test_bert_cls_pooling_ignores_padding(tiny_bert_dir):
    model, tok = load_encoder(tiny_bert_dir, DEVICE)
    assert tok.padding_side == "right" and model.pooler is None
    enc = tokenize(tok, ["river city", "gold tax moon salt king rain"], 16)
    assert enc["input_ids"][0, 0] == tok.cls_token_id and enc["input_ids"][0, 3] == tok.sep_token_id
    assert enc["input_ids"][0, -1] == tok.pad_token_id  # right padded, no extra pad token appended
    hidden = model(**enc).last_hidden_state
    assert torch.allclose(embed(model, enc["input_ids"], enc["attention_mask"], 64), F.normalize(hidden[:, 0]))
    alone = encode(model, tok, ["river city"], DEVICE, 64, 16)
    batched = encode(model, tok, ["river city", "gold tax moon salt king rain bank code"], DEVICE, 64, 16)
    assert torch.allclose(alone[0].float(), batched[0].float(), atol=1e-3)


def test_bert_head_on_last_layernorm(tiny_bert_dir):
    model, tok = load_encoder(tiny_bert_dir, DEVICE)
    enc = tokenize(tok, ["river city", "gold tax"], 16)
    base = model(**enc).last_hidden_state
    attach_head(model, DIM)
    assert torch.allclose(model(**enc).last_hidden_state, model.align_head(base), atol=1e-6)


def test_modernbert_cls_pooling_head_on_final_norm_and_every_weight_trains(tiny_modernbert_dir):
    model, tok = load_encoder(tiny_modernbert_dir, DEVICE)
    assert tok.padding_side == "right" and model.pooling == "cls"
    enc = tokenize(tok, ["river city", "gold tax moon salt king rain"], 16)
    hidden = model(**enc).last_hidden_state
    assert torch.allclose(embed(model, enc["input_ids"], enc["attention_mask"], 64), F.normalize(hidden[:, 0]))
    alone = encode(model, tok, ["river city"], DEVICE, 64, 16)
    batched = encode(model, tok, ["river city", "gold tax moon salt king rain bank code"], DEVICE, 64, 16)
    assert torch.allclose(alone[0].float(), batched[0].float(), atol=1e-3)
    attach_head(model, DIM)
    assert torch.allclose(model(**enc).last_hidden_state, model.align_head(hidden), atol=1e-6)
    model.requires_grad_(True)
    embed(model, enc["input_ids"], enc["attention_mask"], DIM).sum().backward()
    assert all(p.grad is not None for p in model.parameters())  # an unused weight would hang DDP


def test_pooling_read_from_sentence_transformers_config(tiny_mean_bert_dir, tiny_bert_dir):
    assert load_encoder(tiny_bert_dir, DEVICE)[0].pooling == "cls"  # no 1_Pooling/: BERT falls back to CLS
    model, tok = load_encoder(tiny_mean_bert_dir, DEVICE)
    assert model.pooling == "mean"
    enc = tokenize(tok, ["river city", "gold tax moon salt king rain"], 16)
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    assert torch.allclose(embed(model, enc["input_ids"], enc["attention_mask"], 64),
                          F.normalize((hidden * mask).sum(1) / mask.sum(1)), atol=1e-6)
    alone = encode(model, tok, ["river city"], DEVICE, 64, 16)
    batched = encode(model, tok, ["river city", "gold tax moon salt king rain bank code"], DEVICE, 64, 16)
    assert torch.allclose(alone[0].float(), batched[0].float(), atol=1e-3)


@pytest.fixture(scope="module")
def run(tmp_path_factory, tiny_model_dir, tiny_bert_dir):
    """Tiny Qwen teacher index, then 2-process full-FT alignment of the tiny BERT and refinement from it."""
    root = str(tmp_path_factory.mktemp("bert"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    emb_dir = os.path.join(root, "emb")
    build_teacher_index(emb_dir, data, tiny_model_dir, DIM, ("corpus", "align", "val", "test"))
    common = dict(dim=DIM, lr=1e-3, head_lr=1e-2, warmup_ratio=0.0, epochs=2, batch_size=4, log_every=2,
                  eval_every=0, save_every=100, data_path=os.path.join(root, "data.json"), teacher_emb_dir=emb_dir)
    align_cfg = dict(common, model=tiny_bert_dir, full=True, out_dir=os.path.join(root, "align"))
    launch(align_main, align_cfg, nprocs=2)
    refine_cfg = dict(common, init_adapter=os.path.join(align_cfg["out_dir"], "final"),
                      out_dir=os.path.join(root, "refine"))
    launch(refine_main, refine_cfg, nprocs=2)
    return data, emb_dir, align_cfg, refine_cfg


def test_full_align_trains_every_weight(run, tiny_bert_dir):
    _, _, cfg, _ = run
    train = [r["train_loss"] for r in read_log(cfg["out_dir"]) if "train_loss" in r]
    assert len(train) == 6 and train[-1] < train[0]  # 50 align queries / 8 = 6 steps x 2 epochs, logged every 2
    final = os.path.join(cfg["out_dir"], "final")
    assert load_json(os.path.join(final, ADAPTER_CONFIG)) == dict(base=tiny_bert_dir, out_dim=DIM, full=True)
    saved = torch.load(os.path.join(final, ADAPTER_WEIGHTS))
    base, _ = load_encoder(tiny_bert_dir, DEVICE)
    assert set(saved) == set(base.state_dict()) | {"align_head.weight", "align_head.bias"}
    assert not any(".lora_" in k for k in saved)
    w = "encoder.layer.0.attention.self.query.weight"
    assert not torch.allclose(saved[w], base.state_dict()[w])  # the base weights moved, not just the head


def test_refine_from_full_bert(run):
    data, emb_dir, _, cfg = run
    train = [r for r in read_log(cfg["out_dir"]) if "train_loss" in r]
    assert train and all({"cr", "l2"} <= set(r) for r in train)
    docs, _ = load_index(data, emb_dir, ("val",))
    result = evaluate_model(os.path.join(cfg["out_dir"], "final"), data, docs, ("val",), DIM, dtype="fp32")
    assert 0 <= result["val"]["ndcg@10"] <= 1

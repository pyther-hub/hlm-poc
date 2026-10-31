"""Pruning: layer / neuron selection, the model surgery, merging LoRA into the base, full checkpoints, and
prune_main (then refine_main on the pruned student) on 2 CPU processes."""
import json
import os

import pytest
import torch

from align import read_log
from conftest import build_teacher_index, tiny_rows
from data import build_data, save_json
from ddp import launch
from model import (ADAPTER_CONFIG, attach_adapter, encode, load_encoder, merge_lora, prune_model, save_adapter,
                   trainable_params)
from prune import ffn_scores, prune_main, pruned_student, select_layers
from refine import refine_main

DEVICE = torch.device("cpu")
DIM = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TEXTS = ["river city music", "gold tax moon salt king", "rain"]


def vecs(model, tok):
    return encode(model, tok, TEXTS, DEVICE, DIM, 16).float()


def lora_adapter(tiny_model_dir, path):
    """A saved LoRA adapter whose B is non-zero, so it changes the model."""
    model, _ = load_encoder(tiny_model_dir, DEVICE)
    torch.manual_seed(0)
    attach_adapter(model, 4, 8, 0.0, TARGETS, DIM)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(0, 0.05)
    save_adapter(model, path, dict(base=tiny_model_dir, rank=4, alpha=8, dropout=0.0, targets=list(TARGETS),
                                   out_dim=DIM))
    return path


def test_select_layers():
    assert select_layers(28, 4) == [0, 9, 18, 27]
    for n, k in ((28, 14), (14, 7), (7, 4), (4, 2)):
        idx = select_layers(n, k)
        assert len(idx) == k and idx[0] == 0 and idx[-1] == n - 1


def test_prune_keep_all_is_identity(tiny_model_dir):
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    before = encode(model, tok, TEXTS, DEVICE, 64, 16).float()
    prune_model(model, [0, 1], [list(range(128))] * 2)
    assert torch.allclose(encode(model, tok, TEXTS, DEVICE, 64, 16).float(), before, atol=1e-6)


def test_ffn_pruning_drops_dead_neurons(tiny_model_dir):
    """Neurons whose down_proj column is zero score 0, are pruned first, and pruning them changes nothing."""
    model, tok = load_encoder(tiny_model_dir, DEVICE)
    dead = list(range(0, 128, 2))
    with torch.no_grad():
        for layer in model.layers:
            layer.mlp.down_proj.weight[:, dead] = 0
    before = encode(model, tok, TEXTS, DEVICE, 64, 16).float()
    alive = [sorted(ffn_scores(layer.mlp).topk(64).indices.tolist()) for layer in model.layers]
    assert alive == [list(range(1, 128, 2))] * 2
    prune_model(model, [0, 1], alive)
    assert model.layers[0].mlp.up_proj.weight.shape == (64, 64) and model.config.intermediate_size == 64
    assert torch.allclose(encode(model, tok, TEXTS, DEVICE, 64, 16).float(), before, atol=1e-5)


def test_merge_lora_matches_hooks(tiny_model_dir, tmp_path):
    path = lora_adapter(tiny_model_dir, str(tmp_path / "lora"))
    hooked, tok = load_encoder(path, DEVICE)
    merged, _ = load_encoder(tiny_model_dir, DEVICE)
    merge_lora(merged, path)
    assert torch.allclose(vecs(merged, tok), vecs(hooked, tok), atol=1e-5)


def test_pruned_checkpoint_roundtrip_and_chain(tiny_model_dir, tmp_path):
    lora = lora_adapter(tiny_model_dir, str(tmp_path / "lora"))
    model, tok, cfg = pruned_student(lora, 1, 0.5, DEVICE)
    assert cfg["keep_layers"] == [0] and len(cfg["ffn_idx"][0]) == 64 and len(model.layers) == 1
    stage = str(tmp_path / "stage1")
    save_adapter(model, stage, cfg)
    loaded, _ = load_encoder(stage, DEVICE)
    assert torch.allclose(vecs(loaded, tok), vecs(model, tok), atol=1e-6)
    # progressive: the next stage cuts the saved stage further, indices stay relative to the hub base
    _, _, cfg2 = pruned_student(stage, 1, 0.5, DEVICE)
    assert cfg2["keep_layers"] == [0] and len(cfg2["ffn_idx"][0]) == 32
    assert set(cfg2["ffn_idx"][0]) <= set(cfg["ffn_idx"][0])


def test_full_model_trains_every_weight(tiny_model_dir, tmp_path):
    model, _, _ = pruned_student(lora_adapter(tiny_model_dir, str(tmp_path / "lora")), 1, 0.5, DEVICE)
    model.requires_grad_(True)
    body, head = trainable_params(model)
    assert len(head) == 2 and sum(p.numel() for p in body + head) == sum(p.numel() for p in model.parameters())


@pytest.fixture(scope="module")
def run(tmp_path_factory, tiny_model_dir):
    """Tiny teacher index and a LoRA adapter, then 2-process pruning to 1 layer, then refinement of that student."""
    root = str(tmp_path_factory.mktemp("prune"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    emb_dir = os.path.join(root, "emb")
    build_teacher_index(emb_dir, data, tiny_model_dir, DIM, ("corpus", "align", "val"))
    common = dict(dim=DIM, warmup_ratio=0.0, batch_size=4, log_every=3, eval_every=0, save_every=100,
                  data_path=os.path.join(root, "data.json"), teacher_emb_dir=emb_dir)
    prune_cfg = dict(init=lora_adapter(tiny_model_dir, os.path.join(root, "lora")), n_layers=1, ffn_keep=0.5,
                     lr=1e-3, head_lr=1e-3, epochs=2, out_dir=os.path.join(root, "prune"), **common)
    launch(prune_main, prune_cfg, nprocs=2)
    refine_cfg = dict(init_adapter=os.path.join(prune_cfg["out_dir"], "final"), lr=1e-3, head_lr=1e-3, epochs=1,
                      out_dir=os.path.join(root, "refine"), **common)
    launch(refine_main, refine_cfg, nprocs=2)
    return prune_cfg, refine_cfg


def test_prune_main_trains_pruned_student(run):
    prune_cfg, _ = run
    train = [r for r in read_log(prune_cfg["out_dir"]) if "train_loss" in r]
    assert len(train) == 4 and train[-1]["train_loss"] < train[0]["train_loss"]  # 48 align queries / 8 = 6 x 2 steps
    with open(os.path.join(prune_cfg["out_dir"], "final", ADAPTER_CONFIG)) as f:
        cfg = json.load(f)
    assert cfg["full"] and cfg["keep_layers"] == [0] and len(cfg["ffn_idx"][0]) == 64
    model, _ = load_encoder(os.path.join(prune_cfg["out_dir"], "final"), DEVICE)
    assert len(model.layers) == 1


def test_refine_trains_pruned_student(run):
    _, refine_cfg = run
    assert any("cr" in r for r in read_log(refine_cfg["out_dir"]))
    model, _ = load_encoder(os.path.join(refine_cfg["out_dir"], "final"), DEVICE)
    assert len(model.layers) == 1


def test_freeze_embeddings_keeps_them_and_trains_the_rest(run, tiny_model_dir):
    prune_cfg, _ = run
    out_dir = prune_cfg["out_dir"] + "_frozen_emb"
    launch(prune_main, {**prune_cfg, "freeze_embeddings": True, "epochs": 1, "out_dir": out_dir}, nprocs=2)
    base, _ = load_encoder(tiny_model_dir, DEVICE)
    pruned, _ = load_encoder(os.path.join(out_dir, "final"), DEVICE)
    assert torch.equal(pruned.embed_tokens.weight, base.embed_tokens.weight)  # LoRA never touches embeddings
    trained, _ = load_encoder(os.path.join(prune_cfg["out_dir"], "final"), DEVICE)
    assert not torch.equal(trained.embed_tokens.weight, base.embed_tokens.weight)  # unfrozen run does train them
    assert not torch.equal(pruned.norm.weight, base.norm.weight)

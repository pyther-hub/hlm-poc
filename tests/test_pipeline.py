"""Multi-process checks (2 CPU processes, gloo): sharded teacher embedding matches a single process, and is resumable.
The worker is imported from teacher.py (mp.spawn can't pickle test-local functions)."""
import os

import numpy as np
import pytest
import torch

from conftest import tiny_rows
from data import build_data, load_emb, save_json
from ddp import launch
from model import encode, load_encoder
from teacher import QUERY_PROMPT, embed_main

DEVICE = torch.device("cpu")
DIM = 32


@pytest.fixture(scope="module")
def env(tmp_path_factory, tiny_model_dir):
    """Notebook cells 3 + 5: build data, embed it with the (tiny) teacher on 2 processes."""
    root = str(tmp_path_factory.mktemp("pipe"))
    data = build_data(tiny_rows(60), core_n=40, n_val=6, n_test=6, n_align=50, split_seed=1)
    save_json(data, os.path.join(root, "data.json"))
    teacher = dict(model=tiny_model_dir, dim=DIM, dtype="fp32", batch_size=8, doc_max_len=32, query_max_len=32,
                   data_path=os.path.join(root, "data.json"), emb_dir=os.path.join(root, "emb"))
    launch(embed_main, teacher, nprocs=2)
    return data, teacher


def test_sharded_embedding_matches_single_process(env):
    data, teacher = env
    model, tok = load_encoder(teacher["model"], DEVICE)
    for name in ("corpus", "align", "val", "test"):
        emb, ids = load_emb(teacher["emb_dir"], name)
        prefix = "" if name == "corpus" else QUERY_PROMPT
        want = encode(model, tok, data[name]["texts"], DEVICE, DIM, 32, prefix, batch_size=5)
        assert ids == data[name]["ids"] and emb.dtype == np.float16
        assert np.allclose(emb.astype(np.float32), want.float().numpy(), atol=2e-3), name


def test_rerun_skips_finished_sets(env):
    _, teacher = env
    path = os.path.join(teacher["emb_dir"], "corpus.npy")
    before = os.path.getmtime(path)
    launch(embed_main, teacher, nprocs=2)
    assert os.path.getmtime(path) == before

import os
import sys

import numpy as np
import pytest
import torch

# tests never hit the network; tokenizer must already be in the HF cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import clean_rows  # noqa: E402

TINY_TOKENIZER = "openai-community/gpt2"  # small and cached


def tiny_rows(n=60, n_docs=90, seed=0):
    """Fake rows in the raw Tevatron format (then cleaned), with shared and multiple positives."""
    rng = np.random.default_rng(seed)
    words = "river city music cell gold tax moon salt king rain bank code lake film bird wave".split()

    def doc(i):
        return {"docid": f"d{i}", "title": "-", "text": " ".join(rng.choice(words, 12))}
    raw = []
    for q in range(n):
        pos = [doc(int(rng.integers(n_docs))) for _ in range(1 + (q % 3 == 0))]
        raw.append({"query_id": f"q{q}", "query": " ".join(rng.choice(words, 4)) + f" {q}",
                    "positive_passages": pos, "negative_passages": [doc(int(rng.integers(n_docs))) for _ in range(3)]})
    return clean_rows(raw)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """2-layer, hidden-64 random Qwen3 + GPT-2 tokenizer saved as a plain model dir, so load_encoder (and spawned
    embed_main workers) load it like any local checkpoint."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
    path = str(tmp_path_factory.mktemp("tiny_model"))
    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    torch.manual_seed(0)
    Qwen3Model(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def build_teacher_index(emb_dir, data, model_dir, dim, sets):
    """What the teacher notebook writes, without DDP: each set encoded in-process as one shard, merged by
    teacher.merge_shards (so the on-disk layout is the real one)."""
    from model import encode, load_encoder
    from teacher import QUERY_PROMPT, merge_shards, shard_path
    device = torch.device("cpu")
    model, tok = load_encoder(model_dir, device)
    os.makedirs(emb_dir, exist_ok=True)
    for name in sets:
        prefix = "" if name == "corpus" else QUERY_PROMPT
        emb = encode(model, tok, data[name]["texts"], device, dim, 32, prefix)
        np.save(shard_path(emb_dir, name, 0, 1), emb.numpy())
        merge_shards(emb_dir, name, 1, data[name]["ids"])

import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Qwen3Config, Qwen3Model

from ddp import is_main, unwrap

TINY_TOKENIZER = "openai-community/gpt2"  # small and cached; tests + local smoke runs
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tiny_model(vocab_size, seed=0):
    """2-layer, hidden-64 random Qwen3 encoder for tests. No download."""
    config = Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return Qwen3Model(config)


def set_trainable(model, which):
    """which: "all" | "all_but_embeddings" | "none"."""
    for p in model.parameters():
        p.requires_grad = which != "none"
    if which == "all_but_embeddings":
        model.get_input_embeddings().weight.requires_grad = False


def param_counts(model):
    params = list(unwrap(model).parameters())
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)


def load_encoder(name, device, dtype=torch.float32, trainable="none", device_map=None):
    """name: hub id, local checkpoint dir, or "tiny:<seed>" (random test model).
    device_map (e.g. "auto") spreads a model too big for one GPU; device is then ignored."""
    if name.startswith("tiny:"):
        tokenizer = load_tokenizer(TINY_TOKENIZER)
        model = tiny_model(len(tokenizer), seed=int(name[5:]))
    else:
        if name.startswith(("/", ".", "~")) and not os.path.isdir(os.path.expanduser(name)):
            raise FileNotFoundError(f"checkpoint dir not found: {name}")
        tokenizer = load_tokenizer(name)
        model = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation="sdpa", device_map=device_map)
    if device_map is None:
        model = model.to(device=device, dtype=dtype)
    set_trainable(model, trainable)
    return model, tokenizer


def load_encoder_dir(path, device, dtype=torch.float32, trainable="none"):
    """Load a checkpoint written by save_encoder. Fails fast with a clear error if it's missing."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"checkpoint dir not found: {path} (did the stage that writes it run?)")
    return load_encoder(path, device, dtype, trainable)


def save_encoder(model, tokenizer, out_dir, dtype="fp16"):
    """Weights (cast to dtype) + tokenizer. Plain save_pretrained dir, so any later stage can reload it."""
    model = unwrap(model)
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().to(DTYPES[dtype]) for k, v in model.state_dict().items()}
    model.save_pretrained(out_dir, state_dict=state)
    tokenizer.save_pretrained(out_dir)


def tokenize(tokenizer, texts, max_length, prefix=""):
    """Left-padded batch whose last real token is end-of-text, which last-token pooling reads.
    The Qwen3-Embedding tokenizer appends <|endoftext|> itself (also after truncation);
    for tokenizers that don't (GPT-2 in tests) it is appended here."""
    texts = [prefix + t for t in texts]
    if tokenizer("a")["input_ids"][-1] == tokenizer.pad_token_id:
        return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    ids = tokenizer(texts, truncation=True, max_length=max_length - 1)["input_ids"]
    return tokenizer.pad({"input_ids": [x + [tokenizer.pad_token_id] for x in ids]}, return_tensors="pt")


def pool_last_token(hidden, attention_mask):
    # position of the last 1 in each row; works for left and right padding
    last = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def truncate_normalize(x, dim):
    """MRL: keep the first dim features, then L2-normalise."""
    return F.normalize(x[..., :dim].float(), dim=-1)


def embed(model, input_ids, attention_mask, dim):
    hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return truncate_normalize(pool_last_token(hidden, attention_mask), dim)


@torch.no_grad()
def encode(model, tokenizer, texts, device, dim, max_length, prefix="", batch_size=64, amp=True, progress=False):
    """Unit-norm [N, dim] fp16 CPU tensor in input order. Batches are length-sorted so little compute goes to padding."""
    from tqdm.auto import tqdm
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), dim, dtype=torch.float16)
    use_amp = amp and device.type == "cuda" and next(model.parameters()).dtype == torch.float32
    for start in tqdm(range(0, len(texts), batch_size), disable=not (progress and is_main()), desc="encode"):
        idx = order[start:start + batch_size]
        enc = tokenize(tokenizer, [texts[i] for i in idx], max_length, prefix)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            q = embed(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), dim)
        out[idx] = q.half().cpu()
    return out

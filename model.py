import functools
import json
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ddp import is_main

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gpu_mem(device):
    """' (free/total GB free)' on a CUDA device, counting every process on it (e.g. the notebook's own); '' on CPU."""
    if device.type != "cuda":
        return ""
    free, total = torch.cuda.mem_get_info(device)
    return f" ({free / 2**30:.1f}/{total / 2**30:.1f} GB free)"


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_encoder(name, device, dtype=torch.float32, device_map=None):
    """name: hub id, local model dir, or adapter dir (written by save_adapter: base + adapter are rebuilt).
    Frozen, eval mode. device_map (e.g. "auto") spreads a model too big for one GPU; device is then ignored."""
    if name.startswith(("/", ".", "~")) and not os.path.isdir(os.path.expanduser(name)):
        raise FileNotFoundError(f"model dir not found: {name}")
    if os.path.isfile(os.path.join(os.path.expanduser(name), ADAPTER_CONFIG)):
        return load_adapter(name, device, dtype)
    tokenizer = load_tokenizer(name)
    model = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation="sdpa", device_map=device_map)
    if device_map is None:
        model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    return model.eval(), tokenizer


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
    """Unit-norm [N, dim] fp16 CPU tensor in input order. Batches are length-sorted so little compute goes to padding.
    progress=True prints a line every ~1% of batches (rank 0 only), labelled with the device. Plain prints, not tqdm:
    a tqdm bar drawn in a spawned worker doesn't show up in the notebook. Longest texts go first, so the early ETA is
    pessimistic."""
    model.eval()
    #sort in order of token lens for effective gpu batching, then restore original order at the end
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), dim, dtype=torch.float16)
    use_amp = amp and device.type == "cuda" and next(model.parameters()).dtype == torch.float32
    n_batches = -(-len(texts) // batch_size)
    log_every = max(1, n_batches // 100)
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, len(texts), batch_size)):
        idx = order[start:start + batch_size]
        enc = tokenize(tokenizer, [texts[i] for i in idx], max_length, prefix)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            q = embed(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), dim)
        out[idx] = q.half().cpu()
        if progress and is_main() and ((b + 1) % log_every == 0 or b + 1 == n_batches):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (b + 1) * (n_batches - b - 1)
            print(f"[encode {device}] {b + 1}/{n_batches} batches  {elapsed / 60:.1f} min  eta {eta / 60:.1f} min",
                  flush=True)
    return out


# ---------- adapter: LoRA on the decoder's linears + a linear head after the final norm ----------
# Plain parameters registered on the frozen modules + forward hooks; no wrapper classes. Checkpoints hold only the
# adapter: adapter.pt (LoRA + head weights) and adapter_config.json (base model + shapes).

ADAPTER_WEIGHTS, ADAPTER_CONFIG, TRAINER_STATE = "adapter.pt", "adapter_config.json", "trainer_state.pt"


def _lora_hook(scale, dropout, module, args, out):
    x = F.dropout(args[0], dropout, module.training)
    return out + (x @ module.lora_A.T @ module.lora_B.T) * scale


def _head_hook(head, module, args, out):
    return head(out)


def attach_adapter(model, rank, alpha, dropout, targets, out_dim):
    """LoRA (A random, B zero, scale alpha/rank) on every nn.Linear whose name ends in one of `targets`, and a
    Linear(hidden -> out_dim) on the final norm's output, so model(...).last_hidden_state is out_dim wide.
    At init the backbone is unchanged. Adapter params are created trainable, in the base weights' dtype/device."""
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            w = m.weight
            m.lora_A = nn.Parameter(torch.empty(rank, m.in_features, dtype=w.dtype, device=w.device))
            nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
            m.lora_B = nn.Parameter(torch.zeros(m.out_features, rank, dtype=w.dtype, device=w.device))
            m.register_forward_hook(functools.partial(_lora_hook, alpha / rank, dropout))
            n += 1
    assert n, f"no nn.Linear matches lora targets {targets}"
    return attach_head(model, out_dim)


def attach_head(model, out_dim):
    """Linear(hidden -> out_dim) on the final norm's output, so model(...).last_hidden_state is out_dim wide."""
    w = model.norm.weight
    model.align_head = nn.Linear(model.config.hidden_size, out_dim, dtype=w.dtype, device=w.device)
    model.norm.register_forward_hook(functools.partial(_head_hook, model.align_head))
    return model


def is_adapter_key(key):
    return ".lora_" in key or key.startswith("align_head.")


def adapter_params(model):
    """(lora params, head params), trainable."""
    named = [(k, p) for k, p in model.named_parameters() if is_adapter_key(k)]
    return [p for k, p in named if ".lora_" in k], [p for k, p in named if k.startswith("align_head.")]


def trainable_params(model):
    """(body params, head params) among those with requires_grad. With a frozen base the body is the LoRA params,
    in the same order as adapter_params; with a fully trained model it is every base weight."""
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    return [p for k, p in named if not k.startswith("align_head.")], [p for k, p in named if k.startswith("align_head.")]


def merge_lora(model, path):
    """Fold the LoRA of adapter dir `path` into model's base weights (W += alpha/rank * B @ A), no hooks, and attach
    its head. The result is a plain model + head, ready to prune or fully train."""
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        cfg = json.load(f)
    state = torch.load(os.path.join(path, ADAPTER_WEIGHTS), map_location="cpu")
    modules = dict(model.named_modules())
    with torch.no_grad():
        for key in [k for k in state if k.endswith(".lora_A")]:
            w = modules[key[:-len(".lora_A")]].weight
            delta = state[key[:-1] + "B"].float() @ state[key].float() * (cfg["alpha"] / cfg["rank"])
            w += delta.to(w.dtype).to(w.device)
    attach_head(model, cfg["out_dim"])
    model.align_head.load_state_dict({k[len("align_head."):]: v for k, v in state.items()
                                      if k.startswith("align_head.")})
    return model


def prune_model(model, keep_layers, ffn_idx):
    """In place: keep decoder layers keep_layers (indices into model.layers), and in kept layer j only the FFN
    neurons ffn_idx[j] (same count in every layer: Qwen3 has one intermediate_size). Updates the config."""
    layers = [model.layers[i] for i in keep_layers]
    for j, (layer, idx) in enumerate(zip(layers, ffn_idx)):
        mlp = layer.mlp
        idx = torch.as_tensor(idx, device=mlp.up_proj.weight.device)
        for lin, dim in ((mlp.gate_proj, 0), (mlp.up_proj, 0), (mlp.down_proj, 1)):
            lin.weight = nn.Parameter(lin.weight.index_select(dim, idx).clone(), lin.weight.requires_grad)
            lin.in_features, lin.out_features = lin.weight.shape[1], lin.weight.shape[0]
        mlp.intermediate_size = len(idx)
        layer.self_attn.layer_idx = j
    config = model.config
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[i] for i in keep_layers]
    config.num_hidden_layers, config.intermediate_size = len(layers), len(ffn_idx[0])
    model.layers = nn.ModuleList(layers)
    return model


def save_adapter(model, path, adapter_cfg, trainer_state=None):
    """Adapter weights + config (+ optional trainer_state for resuming) into dir `path`, written to path.tmp and
    swapped in, so a crash never leaves a half-written checkpoint. adapter_cfg["full"] = the whole state dict
    (a fully trained, e.g. pruned, model), else only the adapter keys."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if adapter_cfg.get("full") or is_adapter_key(k)}, os.path.join(tmp, ADAPTER_WEIGHTS))
    with open(os.path.join(tmp, ADAPTER_CONFIG), "w") as f:
        json.dump(adapter_cfg, f, indent=1)
    if trainer_state is not None:
        torch.save(trainer_state, os.path.join(tmp, TRAINER_STATE))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def load_adapter_weights(model, path):
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        full = json.load(f).get("full", False)
    missing, unexpected = model.load_state_dict(
        torch.load(os.path.join(path, ADAPTER_WEIGHTS), map_location=next(model.parameters()).device), strict=False)
    missing = [k for k in missing if full or is_adapter_key(k)]
    assert not missing and not unexpected, f"adapter mismatch: missing {missing}, unexpected {unexpected}"


def load_adapter(path, device, dtype=torch.float32):
    """Base model from adapter_config.json + the saved adapter. Frozen, eval mode. A config with keep_layers /
    ffn_idx (indices into the base) is pruned first; full=True has no LoRA, the saved weights replace the base."""
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        cfg = json.load(f)
    model, tokenizer = load_encoder(cfg["base"], device, dtype)
    if "keep_layers" in cfg:
        prune_model(model, cfg["keep_layers"], cfg["ffn_idx"])
    if cfg.get("full"):
        attach_head(model, cfg["out_dim"])
    else:
        attach_adapter(model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["targets"], cfg["out_dim"])
    load_adapter_weights(model, path)
    model.requires_grad_(False)
    return model.eval(), tokenizer

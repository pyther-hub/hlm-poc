# hlm-poc: aligning a small query encoder to a frozen teacher document index

A frozen **teacher** (Qwen3-Embedding-4B) embeds the document corpus once. A small **student** query encoder
(Qwen3-Embedding-0.6B) is trained to retrieve against that frozen index, so documents never need re-embedding when
the query side changes.

**Current scope.** Phase 1: build the MS MARCO data, embed it with the teacher on 2 GPUs, and evaluate retrieval on
val/test against that index. Phase 2: alignment fine-tuning of the student (LoRA r=16 + linear head, DDP) onto the
teacher's query vectors. Phase 3: contrastive refinement of that adapter against the frozen teacher document vectors. Experiments (`run_exp.py`, `experiments.ipynb`): alignment loss, one-shot CL, progressive vs direct pruning (full
fine-tune, `prune.py`), before/after CR, latency and cos(q, d+). Everything runs on Kaggle (GPU T4 x2). This repo holds
the flat, tested functions the notebooks import: plain functions, no classes, no global state, no config system.

---

## 0. Design

### 0.1 Modules
```
 phase_1_teacher.ipynb         phase_1_eval.ipynb         phase_2_align.ipynb     phase_3_refine.ipynb
      │            │                    │                          │                       │
 ┌────▼─────┐ ┌────▼───────┐    ┌───────▼────────┐          ┌──────▼──────┐         ┌──────▼──────┐
 │ data.py  │ │ teacher.py │    │ evaluation.py  │◀─────────│  align.py   │◀────────│  refine.py  │
 │ rows,    │ │ embed_main,│    │ load_index,    │          │ align_main, │         │ refine_main,│
 │ splits,  │ │ merge_     │    │ teacher_ceiling│          │ LOSSES,     │         │ contrastive_│
 │ corpus,  │ │ shards,    │    │ evaluate_model,│          │ run (loop   │         │ loss        │
 │ I/O      │ │ fp16_sanity│    │ summarize,     │          │ shared by   │         │             │
 │          │ │ time_embed.│    │ measure_latency│          │ phases 2+3) │         │             │
 └────┬─────┘ └────┬───────┘    └───────┬────────┘          └──────┬──────┘         └──────┬──────┘
      │       ┌────▼────────────────────▼──────────────────────────▼───────────────────────▼─┐
      └──────▶│ model.py  load_encoder · tokenize · encode · attach_adapter · save/load_adapter│
              └────┬─────────────────────────────────────────────────────────────────────────┘
              ┌────▼──────┐
              │  ddp.py   │  launch/setup/barrier/is_main (no-ops without a process group)
              └───────────┘
```

### 0.2 Data flow
```
Tevatron/msmarco-passage train  ─ stream first 210k rows ─▶ rows_210000.parquet  (query_id, query, positives, negative[0])
   │ make_splits (split_seed)      core 100k → train 90k / val 5k / test 5k;  other 110k
   ▼
build_data ─▶ data.json
   corpus  = positives of core queries + their stored negative, dedup by docid      (~200k passages)
   align   = 90k train + up to 110k other queries, never a val/test query (id or text)  (200k)
   train/val/test = ids, texts, pos (corpus rows), neg (corpus row)
   ▼
embed_main (2 GPUs, shard per rank) ─▶ emb/teacher/{corpus,align,val,test}.npy  fp16 [N, 256] + *_ids.json
   ▼
phase_1_eval (1 GPU): queries (cached teacher vectors, or any model) → top-100 over the teacher corpus
   ─▶ runs/<mode>/eval/metrics.json   {row name: {split: metrics}}
   ▼
align_main (2 GPUs, DDP): student(align query) ≈ teacher align.npy, adapter only
   ─▶ runs/<mode>/align/{step_<n>, final}/ adapter.pt + adapter_config.json, log.jsonl
   ▼
refine_main (2 GPUs, DDP): from align/final; L_CR over train queries vs teacher corpus.npy + λ·L2 to teacher q
   ─▶ runs/<mode>/refine/{step_<n>, final}/, log.jsonl
```

Adapter (`model.attach_adapter`, plain parameters + forward hooks, no wrapper classes):
```
input ─▶ decoder layers (frozen; every q,k,v,o,gate,up,down Linear: W x + B A x · alpha/r) ─▶ final norm
      ─▶ align_head Linear(1024 → 256) ─▶ last-token pool ─▶ L2 normalise      (B = 0 at init)
```
`model(...).last_hidden_state` is already 256 wide, so `embed` / `encode` / `evaluate_model` work unchanged.

DDP pattern:
```
notebook ── launch(embed_main, TEACHER) ──▶ mp.spawn ─┬─ rank 0 (cuda:0): setup → load (downloads) → barrier → embed shard 0
                                                      └─ rank 1 (cuda:1): setup → barrier → load (cache)   → embed shard 1
            barrier → rank 0 merges shards into <set>.npy → barrier → exit
```

### 0.3 Modes
| Mode | Set in the control panel | What changes |
|---|---|---|
| Full | `DEBUG=False` | 210k rows, ~200k passages. Dir `cache/full`. |
| Debug | `DEBUG=True` (default) | Same 4B teacher, ~10k passages. Dir `cache/debug`. |
| Tests | `pytest` | CPU, offline, a random 2-layer Qwen3 saved to a tmp dir (tests only), < 30 s. |

### 0.4 Verified against the Hub (2026-10-27)
| Item | Value | Source |
|---|---|---|
| Teacher query prompt | `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" + q` (no space after `Query:`) | `Qwen3-Embedding-4B/config_sentence_transformers.json` (`prompts.query`) and the model card |
| Passage prompt | none (`prompts.document = ""`) | same file |
| Pooling | last token (`pooling_mode_lasttoken`, `include_prompt: true`), then `Normalize` | `Qwen3-Embedding-0.6B/modules.json`, `1_Pooling/config.json` |
| Padding | left (`padding_side="left"` in the card's examples) | model card |
| End-of-text | both tokenizers **append `<|endoftext|>` (id 151643) themselves**, also after truncation; it is also the pad token, so pooling locates the last token from `attention_mask` | checked locally with `AutoTokenizer` |
| Dimensions / MRL | 4B: 2560, 0.6B: 1024; MRL supports 32..max | model cards |
| Architecture | `qwen3` (`Qwen3ForCausalLM` checkpoint, loaded with `AutoModel`), transformers ≥ 4.51 (we require ≥ 4.56 for the `dtype=` kwarg) | `config.json`, model card |
| Dataset | `Tevatron/msmarco-passage` train: 400.8k rows in one `train.jsonl.gz` (1.7 GB); fields `query_id, query, positive_passages[{docid,title,text}], negative_passages[...]` (~30 negatives per row) | Hub dataset viewer |
| **Not verified** | That `negative_passages` are **BM25** negatives, and their order. The dataset README is empty. We use `negative_passages[0]` and call it the "stored negative". | — |

---

## 1. Using it on Kaggle
Notebook settings: **Accelerator → GPU T4 x2**, **Internet ON**. Open `notebooks/phase_1_teacher.ipynb`. Its first
cell clones the repo (`git pull` on later runs), upgrades `transformers` if it is older than 4.56 (restart the
kernel once if it says so) and installs `datasets`, `pyarrow`, `accelerate` if missing.

Run with `DEBUG = True` first and check that the last cell shows norms ≈ 1.0 and `ids match: True`. Then set
`DEBUG = False`.

| Cell | What it does |
|---|---|
| Control panel | `DEBUG`, `NPROCS`, paths, `DATA`, `TEACHER`, `CHECKS`. |
| 1 · Build data | Stream rows → `build_data` → `data.json` (reused if it already exists). |
| 2 · Pre-flight | `fp16_sanity` (asserts no NaNs) + `time_embedding` (prints projected hours vs `budget_h`). Skipped once `corpus.npy` exists. |
| 3 · Embed | `launch(embed_main, TEACHER, nprocs=2)`. |
| 4 · Inspect | Shape, dtype, norm min/max, file size, and an ids check for every set. |

Sessions are capped at 9 h. Every step is resumable: rows are cached as parquet, `data.json` is reused, and
teacher shards that already exist are skipped.

Then open `notebooks/phase_1_eval.ipynb` with the same `DEBUG` value. It reads the cache built above and needs one GPU.

| Cell | What it does |
|---|---|
| Control panel | `DEBUG`, paths, `EVAL` (splits, k, dim, batch size), `MRL_DIMS`, `MODELS` (name → hub id or checkpoint dir + query prefix/length/dtype), `LATENCY` (same entry shape; the teacher entry and `EVAL["dim"]` are read from `TEACHER_DEFAULTS`). |
| 1 · Load and check | `load_index`: asserts finite, unit-norm vectors, ids in `data.json` order, every positive a valid corpus row. |
| 2 · Teacher ceiling | Row `teacher` (cached teacher queries vs corpus) + `teacher@d64`, `teacher@d128` (MRL sweep). |
| 3 · Models | `evaluate_model` per `MODELS` entry. The untrained student is expected near zero (different embedding space). |
| 4 · Summary | One table per split, teacher first, retention = nDCG@10 / teacher nDCG@10. |
| 5 · Latency | ms/query and queries/s per batch size, teacher (with its prompt) vs student. |

Phase 2: open `notebooks/phase_2_align.ipynb` with the same `DEBUG` value (GPU T4 x2).

| Cell | What it does |
|---|---|
| Control panel | `DEBUG`, `NPROCS`, paths, `ALIGN` (all of `ALIGN_DEFAULTS`, §4.4; `dim` read from `TEACHER_DEFAULTS`). |
| 1 · Check inputs | `check_emb` on the teacher's `align`, `val`, `corpus` vectors; the loss name exists; planned steps. |
| 2 · Train | `launch(align_main, ALIGN, nprocs=NPROCS)`. Resumes from the latest `step_<n>/`; no-op once `final/` exists. |
| 3 · Training log | Train loss and the in-training val rows (val loss, val cos, nDCG@10, Recall@10) from `log.jsonl`. |
| 4 · Evaluate | `evaluate_model(OUT_DIR/final, ...)` on val/test → row `aligned_r<rank>_<loss>` in `runs/<mode>/eval/metrics.json`, then `summarize`. |

Phase 3: open `notebooks/phase_3_refine.ipynb` with the same `DEBUG` value (GPU T4 x2), after phase 2 wrote `align/final`.

| Cell | What it does |
|---|---|
| Control panel | `DEBUG`, `NPROCS`, paths (`INIT_ADAPTER` = `runs/<mode>/align/final`), `REFINE` (all of `REFINE_DEFAULTS`, §4.5). |
| 1 · Check inputs | Adapter config matches `dim`; `check_emb` on `corpus`, `align`, `val`; every train query has a teacher vector. |
| 2 · Train | `launch(refine_main, REFINE, nprocs=NPROCS)`. Resumable, no-op once `final/` exists. |
| 3 · Training log | `loss`, `cr` (contrastive), `l2`, `masked` (fraction of negatives masked), val rows. |
| 4 · Evaluate | `evaluate_model(OUT_DIR/final, ...)` → row `refined_tau<τ>_l2w<λ>` in the shared `metrics.json`, then `summarize`. |

Phase-3 loss, per GPU (B queries) with the global batch's docs (positive + `n_neg` negatives of every query in
it, all ranks; the docs are frozen teacher vectors, so no gather is needed):
```
logits_i = [ s(q_i, d_j) for all (1+n_neg)·B·world docs ] ++ [ s(q_i, q_j) for the B queries on this GPU ]   / tau
masked   = q_i's other labelled positives, q_i itself, and any negative with s > s(q_i, d_i+) + margin
L        = CE(logits, own positive) + l2_weight · ||q_i − teacher(q_i)||²
```

**Loss functions** live in `align.LOSSES` (name → `fn(student, teacher, **loss_kwargs)`, both `[B, dim]` unit
vectors) and are picked in the control panel with `loss` / `loss_kwargs`. They can't be defined in a notebook cell:
the DDP workers are spawned processes and only receive importable functions.

### `experiments.ipynb`
| Cell | What it does |
|---|---|
| Control panel | `DEBUG`, `NPROCS`, `EXP`, inputs (`DATA_PATH`, `EMB_DIR`, `ALIGNED` = phase-2 adapter; on Kaggle under `/kaggle/input/notebooks/tensorpanda231/...`), `EXP_DIR`, `RESUME_FROM`, `ALIGN` + `LOSS_ARMS`, `REFINE`, `PRUNE` + `PRUNE_STAGES` / `PRUNE_DIRECT`, `CR_STUDENTS`, `LATENCY_MODELS` + `LATENCY`, `EVAL`. |
| 1 · Check inputs | Copies `RESUME_FROM` into `EXP_DIR` once; `load_index`; the phase-2 adapter matches `dim`. |
| 2 · Run `EXP` | The `run_exp.exp_*` function: trains missing arms, evaluates missing rows, prints the table, shows the plots. |
| 3 · All rows | `summarize(EXP_DIR, split)` over every experiment's rows. |
| 4 · Saved adapters | Each `EXP_DIR/<arm>/final` with its size, and the plot files. |

Rows / arms: `align_l2`, `align_kl` (losses); `aligned` (input), `aligned_cr`, `oneshot` (oneshot); `prog_l<n>`,
`direct_l<n>` (prune); `<student>_cr` (cr). "P@100" is reported as `recall@100`. The one-shot arm is a proxy:
the teacher (doc tower) stays frozen, unlike the paper's jointly trained baseline.

**Pruning** (`prune.py`): from an adapter dir, LoRA merged into the base (`merge_lora`) or a previous pruned stage;
keep `n_layers` evenly spaced layers (first and last always) and `ffn_keep` of each kept layer's FFN neurons, by
`|gate_j|·|up_j|·|down_:,j|`; then train **every weight** + head with L2 on the align set (`align.run`). The
checkpoint is an adapter dir with `full: true`, `keep_layers`, `ffn_idx` (indices into the hub base) and the whole
state dict in `adapter.pt`; `load_encoder` rebuilds it from the base id. Progressive = chained stages; direct = one
cut with `ffn_keep ** stages` so the final shape matches.

**Corpus caveat:** the index is only the positives + one stored negative of the core queries (~200k passages), not
full MS MARCO (8.8M), so absolute scores are higher than published ones. Use them to compare models.

## 2. Using it locally
```bash
.venv/bin/python -m pytest -q        # CPU, offline, < 30 s
```
The notebook is edited by hand. Commit it without outputs (a test enforces this).

---

## 3. API reference
Docs and notebooks always use `from model import ...` (never `import model`; the name clashes with notebook variables).

### `model.py`
| Function | What it does |
|---|---|
| `load_encoder(name, device, dtype=fp32, device_map=None)` | `name` = hub id, local dir, or adapter dir (→ `load_adapter`). Returns a frozen, eval-mode `(model, tokenizer)`. `AutoModel` with sdpa. A missing local dir raises `FileNotFoundError`. `device_map="auto"` spreads a model over GPUs. |
| `tokenize(tokenizer, texts, max_length, prefix="")` | Left-padded batch ending in end-of-text (appended if the tokenizer doesn't). |
| `pool_last_token(hidden, mask)` / `truncate_normalize(x, dim)` / `embed(model, ids, mask, dim)` | Last-token pooling (any padding side) / MRL truncate + L2 / both after a forward. |
| `encode(model, tokenizer, texts, device, dim, max_length, prefix="", batch_size=64, amp=True, progress=False)` | No-grad, length-sorted batches; fp16 CPU tensor `[N, dim]` in input order. `progress=True`: a `[encode cuda:0] b/n batches  elapsed  eta` line every ~5% (rank 0 only, labelled with the device), plain prints so they show from spawned workers. |
| `get_device()`, `DTYPES` | `cuda:0` or CPU; `"fp32"/"fp16"/"bf16"` → torch dtype. |

### `data.py`
| Function | What it does |
|---|---|
| `load_rows(n_rows, cache_dir)` | Stream the first `n_rows`, `clean_rows` (keep `negative_passages[0]`, drop incomplete rows, dedupe by `query_id`), cache as parquet. |
| `make_splits(n_rows, core_n, n_val, n_test, split_seed)` | Row indices: `train`, `val`, `test` (from the seeded core sample) and `other`. |
| `build_corpus(rows, core_idx, use_title)` | Positives + stored negative of the core queries, deduplicated by docid. |
| `alignment_queries(rows, split, n_align)` | Train + other queries, excluding val/test by id and by text. |
| `build_data(rows, core_n, n_val, n_test, n_align, split_seed, use_title=False)` | All of the above → the `data.json` dict: `corpus`, `align`, `train`/`val`/`test` (`ids`, `texts`, `pos`, `neg`). |
| `load_emb(emb_dir, name)`, `save_json`, `load_json`, `resolve_cfg(defaults, cfg)` | I/O; `resolve_cfg` raises on unknown keys. |

### `teacher.py`
| Function | What it does |
|---|---|
| `embed_main(rank, world_size, cfg)` | Worker for `launch(embed_main, TEACHER)`. Rank r embeds shard r of each set; existing shards are skipped; rank 0 merges. Defaults in `TEACHER_DEFAULTS` (§4.1). |
| `merge_shards(emb_dir, name, world_size, ids)` | Shards → `<name>.npy` + `<name>_ids.json`. |
| `fp16_sanity(model_name, texts, dim, max_len, device, device_map_fp32=None)` | fp16 vs fp32 cosine (mean, min), NaN rows. Use `device_map_fp32="auto"` for the 4B model (16 GB in fp32). |
| `time_embedding(cfg, data, device, world_size, n_sample=2000, budget_h=5.0)` | tokens/s on a passage sample → projected hours for all sets. Report only. |
| `free_gpu_memory()` | `gc.collect()` + `torch.cuda.empty_cache()`. |

### `evaluation.py`
| Function | What it does |
|---|---|
| `load_index(data, emb_dir, splits)` | Teacher corpus + query vectors per split, after `check_emb` (finite, unit norm, ids match) and `check_qrels` (positives valid). Returns `(docs, {split: q})`. |
| `retrieve_topk(queries, docs, k=100, device, chunk=1024)` | Chunked brute-force matmul (fp16 on GPU). |
| `METRICS`, `compute_metrics(topk_idx, qrels)` | Registry: `recall@{1,5,10,20,50,100}`, `mrr@10`, `ndcg@10` (binary gains, multiple positives); mean per metric. |
| `truncate(x, dim)` | MRL: first `dim` features, renormalised. |
| `teacher_ceiling(data, docs, teacher_q, k, device, dims=())` | Rows `teacher` + `teacher@d<dim>` per extra dim. |
| `evaluate_model(model, data, docs, splits, dim, max_query_len, query_prefix, k, device, dtype, batch_size)` | Any hub id / checkpoint dir → `{split: metrics}`. Frees GPU memory after. |
| `save_metrics(out_dir, rows)` / `summarize(out_dir, split, retention_metric="ndcg@10")` | Merge rows into `metrics.json` / print one table, teacher first, with retention. |
| `measure_latency(model, tokenizer, texts, device, dim, max_len, prefix, batch_sizes, repeats=3)` | Median ms/query and queries/s per batch size, with `cuda.synchronize`. |

### `model.py`: adapter
| Function | What it does |
|---|---|
| `attach_adapter(model, rank, alpha, dropout, targets, out_dim)` | LoRA (A Kaiming, B zero) on every `nn.Linear` named in `targets` + `align_head` Linear(hidden → out_dim) after the final norm. Hooks, no wrapper modules. |
| `attach_head(model, out_dim)` | Just the `align_head` (used by `attach_adapter` and by fully trained models). |
| `adapter_params(model)` | `(lora params, head params)`. |
| `trainable_params(model)` | `(body, head)` among `requires_grad` params; body = LoRA for a frozen base, every weight for a full model. `align.run`'s optimizer groups. |
| `merge_lora(model, path)` | Fold an adapter dir's LoRA into the base weights (`W += α/r · B A`) and attach its head; no hooks. |
| `prune_model(model, keep_layers, ffn_idx)` | In place: keep those decoder layers and, per kept layer, those FFN neurons (gate/up rows, down columns); updates `num_hidden_layers`, `intermediate_size`, `layer_types`. |
| `save_adapter(model, path, adapter_cfg, trainer_state=None)` | Dir with `adapter.pt` (adapter weights only, or the whole state dict if `adapter_cfg["full"]`), `adapter_config.json` (`base, rank, alpha, dropout, targets, out_dim`, or `base, out_dim, full, keep_layers, ffn_idx`), optional `trainer_state.pt`. Written to `path.tmp`, then swapped in. |
| `load_adapter(path, device, dtype)` / `load_encoder(path, ...)` | Base from the config (pruned if `keep_layers`), then LoRA + head or, if `full`, head only, then the saved weights; frozen, eval. `load_encoder` dispatches to it when `path` holds `adapter_config.json`. |

### `align.py`
| Function | What it does |
|---|---|
| `align_main(rank, world_size, cfg)` | Worker for `launch(align_main, ALIGN)`. Frozen fp32 base + adapter, DDP over adapter params, AdamW (separate LoRA / head lr), linear warmup + cosine, fp16 autocast + GradScaler on CUDA, grad clip. Each epoch one seeded permutation; rank r takes its slice of every global batch, ragged tail dropped. Logs / evals / saves on rank 0. |
| `LOSSES`, `l2_loss(student, teacher)` | Loss registry; `l2` = squared L2 between unit vectors (= 2 − 2 cos). |
| `run(model, tokenizer, adapter_cfg, cfg, rank, world_size, device, n_items, batch_loss, data, tag)` | The loop shared by phases 2 and 3: optimizer, schedule, amp, resume, DDP, log / eval / save, `final/`. `batch_loss(ddp_model, global_idx, local, epoch)` returns `{"loss", ...logged parts}`. |
| `load_rank0_first`, `final_exists`, `load_val`, `evaluate_val`, `save_checkpoint`, `log_row` | Loop helpers: rank-0-first download, no-op check, in-training val (L2, cos, nDCG@10, Recall@10), step checkpoint + `keep_last`, `log.jsonl`. |
| `step_checkpoints(out_dir)`, `read_log(out_dir)` | Finished `step_<n>` dirs, oldest first / rows of `log.jsonl`. |

### `refine.py`
| Function | What it does |
|---|---|
| `refine_main(rank, world_size, cfg)` | Worker for `launch(refine_main, REFINE)`. Loads `init_adapter` (base + adapter), unfreezes the adapter (every weight for a `full` pruned init), trains the train split with `align.run`. Writes only to `out_dir`. |

### `prune.py`
| Function | What it does |
|---|---|
| `prune_main(rank, world_size, cfg)` | Worker for `launch(prune_main, PRUNE)`. `pruned_student`, unfreeze everything, L2 on the align set with `align.run`. Defaults in `PRUNE_DEFAULTS` (§4.6). |
| `pruned_student(init, n_layers, ffn_keep, device)` | `(model, tokenizer, adapter_cfg)`: init adapter merged (LoRA) or loaded (pruned stage), cut, with indices relative to the hub base. |
| `select_layers(n, k)`, `ffn_scores(mlp)`, `top_neurons(mlp, k)` | Evenly spaced layers (first, last kept); per-neuron `|gate_j|·|up_j|·|down_:,j|`; the top k, sorted. |

### `run_exp.py`
| Function | What it does |
|---|---|
| `exp_losses(align_cfg, losses, data, index, exp_dir, ev, nprocs)` | Phase 2 per arm `{arm: (loss, kwargs)}`; curves + bars + table. |
| `exp_oneshot(align_cfg, refine_cfg, aligned, data, index, exp_dir, ev, nprocs)` | Fresh adapter → `refine_main` with `l2_weight=0` for ≈ align + CR steps, vs `aligned` and `aligned_cr`. |
| `exp_prune(prune_cfg, aligned, stages, direct, data, index, exp_dir, ev, nprocs)` | Progressive `prog_l<n>` chain and direct `direct_l<n>`; metric vs layers plot. |
| `exp_cr(refine_cfg, students, data, index, exp_dir, ev, nprocs)` | `<student>_cr` per student; gains printed, bars + curves. |
| `exp_latency(models, data, index, exp_dir, device, dim, n_queries, batch_sizes)` | `measure_latency` per model (first = reference) → `latency.json`, speedups; mean cos(q, d+) on val; plot. |
| `kl_loss(student, teacher, tau)`, `exp_align_main` | KaLM-style KL over the batch's teacher vectors; `align_main` with `"kl"` registered in the worker (L2 stays the default elsewhere). |
| `train_once`, `evaluate_runs`, `cos_to_positive`, `plot_curves`, `plot_bars`, `print_rows`, `save_fig` | Skip finished arms / rows (+ the teacher row); helpers. |
| `mine_negatives(teacher_q, docs, pos, neg, n_neg, chunk=256)` | `[N, n_neg]` negative corpus rows per query: stored negative first, then the top teacher-scored docs excluding positives and the stored one. `n_neg=1` skips scoring. |
| `contrastive_loss(q, docs, pos_col, known_pos, tau, margin)` | L_CR with doc + same-tower query negatives and false-negative masking; returns `(loss, masked fraction)`. |

### `ddp.py`
`launch(fn, *args, nprocs)`, `setup`, `cleanup`, `barrier`, `is_main`, `get_rank`, `get_world_size`,
`reduce_mean`, `wrap_ddp`, `unwrap`. Copied from tiny-reasoner; all are no-ops without a process group.

---

## 4. Config keys

### 4.1 `TEACHER` (`teacher.TEACHER_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `model` | `"Qwen/Qwen3-Embedding-4B"` | Hub id or local dir. Same in debug and full. |
| `dim`, `dtype` | `256`, `"fp16"` | MRL dim; teacher weights dtype. |
| `query_prompt` | `QUERY_PROMPT` | Queries only. |
| `doc_max_len` / `query_max_len` | `128` / `64` | Tokens (query length includes the ~20-token prompt). |
| `batch_size` | `64` | Per GPU. |
| `data_path` / `emb_dir` | `cache/data.json` / `cache/emb/teacher` | The notebook sets them under `cache/{debug,full}`. |
| `sets` | `("corpus", "align", "val", "test")` | Embedded and merged. |

### 4.2 `DATA` (control panel)
`n_rows=210_000`, `core_n=100_000`, `n_val=5_000`, `n_test=5_000`, `n_align=200_000`, `split_seed=1234`,
`use_title=False`. Debug: `12_000 / 5_000 / 500 / 500 / 10_000` (~10k passages).

### 4.3 `CHECKS` (control panel)
`sanity_n=1000` passages for the fp16 check, `timing_n=2000` passages timed, `budget_h=5.0` (reported only).

### 4.4 `ALIGN` (`align.ALIGN_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `model` | `"Qwen/Qwen3-Embedding-0.6B"` | Student base, frozen, loaded fp32. |
| `dim` | `256` | Head output = teacher index dim. |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `16`, `32`, `0.05` | LoRA scale = alpha / rank. |
| `lora_targets` | `q,k,v,o,gate,up,down _proj` | Every linear in the decoder. |
| `query_prefix`, `max_query_len` | `""`, `32` | Student queries. |
| `loss`, `loss_kwargs` | `"l2"`, `{}` | Name in `align.LOSSES` + its kwargs. |
| `lr`, `head_lr`, `weight_decay`, `warmup_ratio` | `2e-4`, `1e-3`, `0.0`, `0.05` | AdamW; warmup then cosine to 0. |
| `epochs`, `batch_size`, `grad_clip` | `1`, `128`, `1.0` | `batch_size` per GPU. |
| `amp`, `seed` | `True`, `0` | fp16 autocast + GradScaler on CUDA; data order and adapter init. |
| `log_every`, `eval_every`, `save_every`, `keep_last` | `20`, `200`, `500`, `2` | Steps. Debug: `5 / 20 / 20`. |
| `data_path`, `teacher_emb_dir`, `out_dir` | `cache/data.json`, `cache/emb/teacher`, `runs/align` | The notebook sets them under `{debug,full}`. |

### 4.5 `REFINE` (`refine.REFINE_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `init_adapter` | `"runs/align/final"` | Phase-2 adapter dir; read only. LoRA shape and dropout come from its config. |
| `dim`, `query_prefix`, `max_query_len` | `256`, `""`, `32` | Must match phase 2 / the teacher index. |
| `tau`, `margin`, `l2_weight` | `0.05`, `0.1`, `0.1` | Temperature; false-negative margin; λ of the L2 term (0 = pure L_CR). |
| `n_neg` | `1` | Negatives per query. `1` = the stored `negative[0]` only (no mining); `k > 1` adds the `k-1` corpus docs the teacher scores highest for the query (never a labelled positive), mined once at start. |
| `lr`, `head_lr`, `weight_decay`, `warmup_ratio` | `5e-5`, `1e-4`, `0.0`, `0.05` | AdamW; warmup then cosine to 0. |
| `epochs`, `batch_size`, `grad_clip`, `amp`, `seed` | `1`, `64`, `1.0`, `True`, `0` | `batch_size` = queries per GPU (docs per step = (1 + n_neg) · batch_size · world_size). |
| `log_every`, `eval_every`, `save_every`, `keep_last` | `20`, `200`, `500`, `2` | Steps. Debug: `5 / 20 / 20`. |
| `data_path`, `teacher_emb_dir`, `out_dir` | `cache/data.json`, `cache/emb/teacher`, `runs/refine` | The notebook sets them under `{debug,full}`. |

### 4.6 `PRUNE` (`prune.PRUNE_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `init` | `"runs/align/final"` | Phase-2 adapter or a previous pruned stage; read only. |
| `n_layers`, `ffn_keep` | `14`, `0.75` | Layers kept (evenly spaced over the init's); fraction of the init's FFN width kept. |
| `dim`, `query_prefix`, `max_query_len` | `256`, `""`, `32` | Must match phase 2 / the teacher index. |
| `lr`, `head_lr`, `weight_decay`, `warmup_ratio` | `2e-5`, `1e-4`, `0.0`, `0.05` | AdamW over every weight / the head; warmup then cosine to 0. |
| `epochs`, `batch_size`, `grad_clip`, `amp`, `seed` | `1`, `64`, `1.0`, `True`, `0` | `batch_size` per GPU. |
| `log_every`, `eval_every`, `save_every`, `keep_last` | `20`, `200`, `500`, `2` | Steps. Debug: `5 / 20 / 20`. |
| `data_path`, `teacher_emb_dir`, `out_dir` | `cache/data.json`, `cache/emb/teacher`, `runs/prune` | Set per arm by `run_exp.exp_prune`. |

---

## 5. Tests (`tests/`, CPU, offline, < 30 s)
| Test | Catches |
|---|---|
| `test_pooling_padding_invariant`, `test_tokenize_ends_with_eot` | Pooling depends on padding; pooled token isn't end-of-text. |
| `test_truncate_normalize_unit_norm`, `test_encoder_is_frozen`, `test_missing_model_dir_fails_fast` | Non-unit vectors; teacher not frozen; silent missing dir. |
| `test_splits_disjoint_by_query`, `test_corpus_has_positives_and_negatives`, `test_alignment_queries_exclude_val_test` | Leakage between splits, missing positives/negatives, val/test queries in alignment. |
| `test_resolve_cfg_rejects_typos` | Typos in a control panel silently ignored. |
| `test_sharded_embedding_matches_single_process` | 2-process sharded teacher embedding ≠ single process. |
| `test_rerun_skips_finished_sets` | A re-run re-embeds sets that are already done. |
| `test_retrieve_topk_matches_numpy`, `test_compute_metrics_by_hand`, `test_truncate_renormalizes` | Retrieval / metric / MRL bugs. |
| `test_check_emb_rejects_bad_inputs` | Sanity checks that don't catch bad norms or misaligned ids. |
| `test_eval_flow` | The eval notebook's flow on a tiny index; the teacher model through `evaluate_model` reproduces the cached ceiling. |
| `test_measure_latency_runs` | Latency helper. |
| `test_adapter_starts_as_base_plus_head`, `test_adapter_checkpoint_roundtrip` | LoRA changing the model at init; checkpoints holding base weights or not reloading to the same vectors. |
| `test_l2_loss_is_two_minus_two_cos` | Loss scale. |
| `test_align_trains_and_keeps_last_checkpoints`, `test_align_resumes_from_latest_checkpoint` | 2-process alignment: loss goes down, `keep_last`, final adapter evaluates; a re-run resumes mid-run instead of restarting. |
| `test_contrastive_loss_by_hand`, `test_contrastive_loss_masks_negatives_above_margin` | L_CR formula, known-positive masking, margin masking (a masked doc = an absent doc). |
| `test_mine_negatives`, `test_refine_with_mined_negatives` | Mined negatives skip positives and the stored negative (`n_neg=1` = stored only); a 2-process run with `n_neg=3` trains and saves. |
| `test_refine_trains_from_adapter` | 2-process refinement from a saved adapter: contrastive loss goes down, val rows logged, the init adapter untouched, final adapter evaluates. |
| `test_select_layers`, `test_prune_keep_all_is_identity`, `test_ffn_pruning_drops_dead_neurons` | Layer choice; surgery that changes outputs when it shouldn't; neuron scores / slicing. |
| `test_merge_lora_matches_hooks`, `test_pruned_checkpoint_roundtrip_and_chain`, `test_full_model_trains_every_weight` | Merged LoRA ≠ hooked LoRA; full checkpoints not reloading; progressive indices not relative to the base; frozen weights in a full fine-tune. |
| `test_prune_main_trains_pruned_student`, `test_refine_trains_pruned_student` | 2-process pruning trains and saves a 1-layer full checkpoint; CR on it trains. |
| `test_kl_loss_zero_iff_same`, `test_lora_trainable_params_match_adapter_params`, `test_evaluate_runs_adds_teacher_and_resumes` | KL scale; LoRA optimizer groups changed (breaks resume); finished rows re-evaluated; plots. |
| `test_expected_notebooks`, `test_notebook_clean_and_compiles` | Stray notebooks, stored outputs, syntax errors. |

## 6. Workflow
1. Edit locally → `.venv/bin/python -m pytest -q` → commit → push.
2. On Kaggle: re-run the setup cell, run with `DEBUG = True`, then `DEBUG = False`.
3. After the teacher index exists: run `phase_1_eval` with the same `DEBUG` value.
4. Then `phase_2_align` (same `DEBUG`); its final adapter is scored into the same `metrics.json`.
5. Then `phase_3_refine` (same `DEBUG`), which starts from `align/final`; its row lands in the same table.
6. Then `experiments`, one `EXP` per session (`losses`, `oneshot`, `prune`, then `cr` and `latency`, which reuse the
   pruned arms). Save the version and set `RESUME_FROM` to its output in the next session.

# hlm-poc: aligning a small query encoder to a frozen teacher document index

3-day proof of concept of the Harness-LM idea. A frozen **teacher** (Qwen3-Embedding-4B) embeds the document
corpus once. A small **student** query encoder (Qwen3-Embedding-0.6B) is trained to retrieve against that frozen
index, so documents never need re-embedding when the query side changes.

Experiments run in three Kaggle notebooks (GPU T4 x2, DDP). This repo holds the flat, tested functions they import,
in the same style as `tiny-reasoner`: plain functions, notebooks as orchestrators, no classes, no global state, no
config system. Pruning is out of scope (save/load is a plain `save_pretrained` dir, so it can be added later).

---

## 0. High-level design

### 0.1 Idea in one picture
```
                  embed once (frozen, fp16, d=256)
MS MARCO corpus ──────────────────────────────────▶  teacher doc index  [~200k x 256]
                                                          ▲
teacher queries ── teacher (+ instruction prompt) ──▶ teacher query vecs │ retrieve top-100
                                                          │              │
student queries ── student (no prompt) ─────────────▶ student query vecs ┘
                      ▲
                      │  stage A  "align":  L2(student q, teacher q)            → aligned/
                      │  stage C  "CR":     InfoNCE vs frozen teacher doc vecs   → cr_only/, aligned_cr/
```

### 0.2 System context
```
┌──────────────────────────────┐   git push / pull      ┌────────────────────────────────────┐
│  Local Mac (CPU)             │ ─────────────────────▶ │  Kaggle notebook (GPU T4 x2)       │
│  edit → pytest → commit      │       GitHub repo      │  setup cell: clone/pull, sys.path  │
│  TINY-mode notebook runs     │                        │  launch(worker, cfg) on both GPUs  │
└──────────────────────────────┘                        └─────────────────┬──────────────────┘
                                         HF Hub: Qwen3-Embedding-{4B,0.6B}, Tevatron/msmarco-passage
                         /kaggle/working/cache/<mode>/  (rows parquet, data.json, teacher .npy)
                         /kaggle/working/runs/<mode>/   (baselines/, aligned/, cr_only/, aligned_cr/)
```

### 0.3 Modules and dependencies
```
            notebooks (control panels)  ──  00_baseline · 01_align_only · 02_cr_only
                 │            │              │                 │
          ┌──────▼───┐  ┌─────▼─────┐  ┌─────▼──────┐  ┌───────▼───────┐
          │ data.py  │  │teacher.py │  │  train.py  │  │ evaluation.py │
          │ rows,    │  │ embed_main│  │ train_loop │  │ retrieve_topk │
          │ splits,  │  │ (sharded) │  │ align_main │  │ metrics       │
          │ corpus,  │  │ sanity,   │  │ cr_main    │  │ evaluate_dir  │
          │ loaders  │  │ timing    │  │ scaling    │  │ summarize_runs│
          └────┬─────┘  └─────┬─────┘  └──┬──────┬──┘  └───────┬───────┘
               │              │           │ losses.py (LOSSES)  │
          ┌────▼──────────────▼───────────▼─────────────────────▼──┐
          │ model.py  load_encoder · tokenize · pool · encode · save│
          └────────────────────────────┬────────────────────────────┘
                                ┌──────▼──────┐
                                │   ddp.py    │  launch/setup/is_main/reduce_mean/wrap_ddp/unwrap
                                └─────────────┘  (no-ops without a process group; copied from tiny-reasoner)
```
| Module | Responsibility |
|---|---|
| `model.py` | Load an encoder (hub id, checkpoint dir, or `tiny:<seed>`), left-padded tokenization ending in end-of-text, last-token pooling, MRL truncate + normalise, batched `encode`, save/load. |
| `data.py` | Stream and cache MS MARCO rows, split by query, build corpus / query sets / alignment queries / CR triples, collate + loaders, JSON and embedding I/O, `resolve_cfg`. |
| `teacher.py` | Data-parallel teacher embedding (`embed_main`, resumable per shard), `merge_shards`, `fp16_sanity`, `time_embedding` go/no-go. |
| `losses.py` | `LOSSES` registry: `l2`, `cosine`, `contrastive`, `hybrid`. One short file; its docstring shows how to add one. |
| `train.py` | Stage defaults, `train_loop` (one explicit function), `run_stage`, `align_main`, `cr_main`, `LR_SCHEDULES`, `benchmark_scaling`, checks. |
| `evaluation.py` | Brute-force top-k, `METRICS` registry, `teacher_ceiling`, `evaluate_dir`, `summarize_runs`, `measure_latency`. |
| `ddp.py` | Thin `torch.distributed` helpers, identical to tiny-reasoner's. |

### 0.4 Data flow
```
Tevatron/msmarco-passage train  ─ stream first 210k rows ─▶ rows_210000.parquet  (query_id, query, positives, negative[0])
   │ make_splits (split_seed)      core 100k → train 90k / val 5k / test 5k;  other 110k
   ▼
build_data ─▶ data.json
   corpus  = positives of core queries + their stored negative, dedup by docid      (~200k passages)
   align   = 90k train + up to 110k other queries, never a val/test query (id or text)  (200k)
   train/val/test = ids, texts, pos (corpus rows), neg (corpus row)  → CR triples = (query, pos[0], neg)
   ▼
embed_main (2 GPUs, shard per rank) ─▶ emb/teacher/{corpus,align,val,test}.npy  fp16 [N, 256] + *_ids.json
   ▼
align_main: student(q) vs teacher q vec ──── LOSSES["l2"] ─▶ aligned/
cr_main:    student(q) vs teacher doc table (pos, hard neg, in-batch, same-tower, random) ─▶ cr_only/, aligned_cr/
   ▼
evaluate_dir: student(val/test q) → top-100 over teacher corpus → metrics.json in the checkpoint dir
summarize_runs: one table, teacher row first
```
Stages communicate **only through directories** (weights, `cfg.json`, `train_log.jsonl`, `metrics.json`), so any
stage can be re-run from any checkpoint.

### 0.5 Execution modes
| Mode | Set in the control panel | What changes |
|---|---|---|
| Full | `DEBUG=False` | 4B teacher, 210k rows, full loops. Dirs `cache/full`, `runs/full`. |
| Debug | `DEBUG=True` (default) | 0.6B debug teacher via the same code path, ~10k passages, 3 steps per loop, shape printing, grad check, weights-in-sync check. Dirs `*/debug`. |
| Tiny | `TINY=True` | Random tiny Qwen3 models + fake rows, CPU, 2 gloo processes. Dirs `*/tiny`. For local smoke runs. |
| Tests | `pytest` | CPU, offline, tiny models, < 30 s. |

DDP pattern (both training stages and teacher embedding):
```
notebook ── launch(worker, cfg) ──▶ mp.spawn ─┬─ rank 0 (cuda:0): setup → load (downloads) → barrier → ... → save
                                              └─ rank 1 (cuda:1): setup → barrier → load (cache)   → ...
   training: DistributedSampler + set_epoch, no_sync on non-update accumulation steps, reduce_mean for logs,
             rank-0-only printing/saving, barrier before exit
   teacher:  rank r embeds shard r, existing shards skipped, rank 0 merges
notebook ◀── reads out_dir (evaluation is single-process, in 00)
```

### 0.6 Key design decisions
- **Frozen teacher index, embedded once.** The student never sees a document encoder: CR uses the cached teacher doc
  vectors as a lookup table, loaded once per rank onto its own GPU (~100 MB).
- **Alignment needs no teacher forward pass.** Targets are the cached teacher query vectors.
- **Token embeddings frozen by default** (`trainable="all_but_embeddings"`): ~0.44B of 0.6B params train; fp32
  master weights + AdamW ≈ 7 GB per T4.
- **fp16 autocast + GradScaler** (T4 has no bf16). `amp=False` is the plain fp32 fallback if NaNs appear;
  a non-finite loss always raises with that hint.
- **Every result-affecting knob lives in the notebook control panel**, merged over the stage's defaults dict by
  `resolve_cfg` (unknown keys raise). The resolved config is printed before training and saved as `cfg.json`.
- **Registries** for losses (`LOSSES`), lr schedules (`LR_SCHEDULES`) and metrics (`METRICS`): adding one is a few lines.

### 0.7 Verified against the Hub (2026-10-27)
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
Notebook settings: **Accelerator → GPU T4 x2**, **Internet ON**. Each notebook starts with this setup cell:
```python
REPO_URL = "https://github.com/pyther-hub/hlm-poc.git"   # clone on first run, git pull afterwards
```
It also upgrades `transformers` if it is older than 4.56 (restart the kernel once if it says so) and installs
`datasets`, `pyarrow`, `accelerate` if missing.

Order: **00** (cells 1–4) → **01** → **02** → **00** last cell. Run every notebook with `DEBUG = True` first, then
flip to `False`. Debug and full runs use separate `cache/<mode>` and `runs/<mode>` dirs.

Sessions are capped at 9 h. Long steps are resumable: rows are cached as parquet, `data.json` is reused, teacher
shards that exist are skipped, every stage writes to its own dir.

## 2. Using it locally
```bash
.venv/bin/python -m pytest -q                       # ~25 s, CPU, offline
.venv/bin/python scripts/make_notebooks.py          # regenerate notebooks after editing their cells
.venv/bin/python scripts/run_notebooks_tiny.py      # execute 00 → 01 → 02 → 00 in TINY mode (~1 min)
```
The notebooks are generated by `scripts/make_notebooks.py`. Edit cells there, not in the `.ipynb`.

---

## 3. Notebooks

### 00_baseline.ipynb: baselines and evaluation (no training)
| Cell | Does |
|---|---|
| Control panel | `PATHS`, `MODELS`, `DIM`, `DATA`, `TEACHER`, `CHECKS`, `EVAL`, `RUN_NATIVE`, `RUN_LATENCY`, `EVAL_DIRS` |
| 1 · Build data | `load_rows` → `build_data` → `data.json` (prints one row so you can inspect it) |
| 2 · Teacher | `fp16_sanity` (fp16 vs fp32 cosine on 1000 passages, NaN check) → `time_embedding` (go/no-go: over 3 h halves the corpus and alignment queries, then rebuilds) → `launch(embed_main, TEACHER)` |
| 3 · Teacher ceiling | `teacher_ceiling` → `baselines/metrics.json["teacher"]` |
| 4 · Untrained student | `evaluate_dir(student)` against the teacher index. **Expected near zero** (different embedding spaces); also prints mean cos(student q, teacher q) on val |
| 5 · Native (optional) | Student in its own space (own corpus embedding, own query prompt), reference row only |
| Latency (optional) | `measure_latency` of the student |
| 6 · Summary | `evaluate_dir` on every trained dir in `EVAL_DIRS`, then `summarize_runs` → one table, teacher row first. **Re-run after each training notebook.** |

### 01_align_only.ipynb: stage A
Control panel `ALIGN_CFG` (all keys, §5.1). Cells: **plan** (`plan_only=True`: resolved cfg, effective batch,
optimizer updates, trainable/total params, first-batch shapes) → **scaling** (`benchmark_scaling`, 1 vs 2 GPUs) →
**debug run** → **real run** → **what was saved**. Output: `runs/<mode>/aligned/`.

### 02_cr_only.ipynb: stage C, two runs
One shared `CR_CFG` plus `RUN_A` / `RUN_B`, which differ only in `init` and `out_dir`:
- Run A: `init` = pretrained student → `cr_only/`
- Run B: `init` = `runs/<mode>/aligned` → `aligned_cr/` (raises right away with a clear error if `aligned/` is missing)

Cells: plan → debug run → Run A → Run B → what was saved. Evaluate both in 00.

---

## 4. API reference
Docs and notebooks always use `from model import ...` (never `import model`; the name clashes with notebook variables).

### `model.py`
| Function | What it does |
|---|---|
| `load_encoder(name, device, dtype=fp32, trainable="none", device_map=None)` | `name` = hub id, checkpoint dir, or `"tiny:<seed>"`. Returns `(model, tokenizer)`. `AutoModel` with sdpa. A missing local dir raises `FileNotFoundError`. `device_map="auto"` spreads a model over GPUs. |
| `load_encoder_dir(path, device, ...)` | Same, but requires `path` to be an existing dir. |
| `save_encoder(model, tokenizer, out_dir, dtype="fp16")` | `save_pretrained` (weights cast to dtype) + tokenizer. Unwraps DDP. |
| `set_trainable(model, which)` / `param_counts(model)` | `all` / `all_but_embeddings` / `none`; (trainable, total). |
| `tokenize(tokenizer, texts, max_length, prefix="")` | Left-padded batch ending in end-of-text (appended if the tokenizer doesn't). |
| `pool_last_token(hidden, mask)` / `truncate_normalize(x, dim)` / `embed(model, ids, mask, dim)` | Last-token pooling (any padding side) / MRL truncate + L2 / both after a forward. |
| `encode(model, tokenizer, texts, device, dim, max_length, prefix="", batch_size=64, amp=True)` | No-grad, length-sorted batches; fp16 CPU tensor `[N, dim]` in input order. |
| `tiny_model(vocab_size, seed=0)` | 2-layer, hidden-64 random Qwen3 for tests. |

### `data.py`
| Function | What it does |
|---|---|
| `load_rows(n_rows, cache_dir)` | Stream the first `n_rows`, `clean_rows` (keep `negative_passages[0]`, drop incomplete rows, dedupe by `query_id`), cache as parquet. |
| `make_splits(n_rows, core_n, n_val, n_test, split_seed)` | Row indices: `train`, `val`, `test` (from the seeded core sample) and `other`. |
| `build_corpus(rows, core_idx, use_title)` | Positives + stored negative of the core queries, deduplicated by docid. |
| `alignment_queries(rows, split, n_align)` | Train + other queries, excluding val/test by id and by text. |
| `build_data(rows, core_n, n_val, n_test, n_align, split_seed, use_title=False)` | All of the above → the `data.json` dict: `corpus`, `align`, `train`/`val`/`test` (`ids`, `texts`, `pos`, `neg`). |
| `cr_triples(data)` | `(query, pos[0], neg)` per train query. |
| `collate_queries(batch, tokenizer, max_len)` / `make_loader(items, tokenizer, batch_size, max_len, shuffle, distributed, seed, drop_last)` | Items are dicts with `text` and optional `target`, `pos`, `neg`. |
| `load_emb(emb_dir, name)`, `save_json`, `load_json`, `resolve_cfg(defaults, cfg)` | I/O; `resolve_cfg` raises on unknown keys, replaces dict values whole. |
| `tiny_rows(n)` | Fake rows in the raw Tevatron format (shared + multiple positives). |

### `teacher.py`
| Function | What it does |
|---|---|
| `embed_main(rank, world_size, cfg)` | Worker for `launch(embed_main, TEACHER)`. Rank r embeds shard r of each set; existing shards are skipped; rank 0 merges. Defaults in `TEACHER_DEFAULTS` (§5.3). |
| `merge_shards(emb_dir, name, world_size, ids)` | Shards → `<name>.npy` + `<name>_ids.json`. |
| `fp16_sanity(model_name, texts, dim, max_len, device, device_map_fp32=None)` | fp16 vs fp32 cosine (mean, min), NaN rows. Use `device_map_fp32="auto"` for the 4B model (16 GB in fp32). |
| `time_embedding(cfg, data, device, world_size, n_sample=2000, budget_h=3.0)` | tokens/s on a passage sample → projected hours for all sets → `go`. |
| `halve_sizes(data_cfg)` | Halve the train part of the core set and `n_align` (val/test unchanged). |

### `losses.py`
`LOSSES[name](q, batch, **loss_kwargs)`. `q` is the student's unit-norm query vectors `[B, d]`.
| Loss | What it does |
|---|---|
| `l2` | Mean squared L2 to the teacher query vector (unit norm, so `2 − 2cos`). |
| `cosine` | `1 − cos`. |
| `contrastive(tau=0.05, n_hard_neg=1, same_tower=True, mask_margin=0.1, extra_negs=0)` | Qwen3-style InfoNCE: in-batch positives, the stored hard negative, other queries (same tower), `extra_negs` random corpus docs. Masks a candidate if it is the query's positive doc or scores > positive + `mask_margin`. Negatives are per rank (no all_gather). |
| `hybrid(terms)` | Weighted sum, e.g. `terms={"l2": {"weight": 1}, "contrastive": {"weight": 0.1, "tau": 0.05}}`. |

### `train.py`
| Function | What it does |
|---|---|
| `align_main(rank, world_size, cfg)` / `cr_main(...)` | Stage workers for `launch`. Merge `cfg` over `ALIGN_DEFAULTS` / `CR_DEFAULTS`. |
| `run_stage(rank, world_size, cfg, make_data, prepare)` | Shared body: setup → model → loaders → AdamW + schedule → plan → (plan_only / benchmark / train) → save on rank 0. |
| `train_loop(model, loader, optimizer, scheduler, device, cfg, prepare, val_fn=None)` | The loop: batch → forward → `LOSSES[cfg["loss"]]` → backward → step. Accumulation with `no_sync`, GradScaler, grad clipping, logging every `val_every` updates. |
| `LR_SCHEDULES` | `cosine` (warmup + cosine to `lr × min_lr_ratio`), `constant` (warmup then flat). |
| `benchmark_scaling(worker, cfg, steps=20, nprocs_list=(1, 2))` | samples/s per process count + scaling efficiency; warm-up discarded, nothing saved. |
| `check_grads`, `assert_ranks_in_sync`, `num_update_steps`, `clean_unused_gpu_memory` | Checks and utilities. |

### `evaluation.py`
| Function | What it does |
|---|---|
| `retrieve_topk(queries, docs, k=100, device, chunk=1024)` | Chunked brute-force matmul (fp16 on GPU). |
| `compute_metrics(topk_idx, qrels)` | Mean of every `METRICS` entry: `recall@100`, `mrr@10`, `ndcg@10` (binary gains, multiple positives). |
| `teacher_ceiling(data_path, emb_dir, out_dir, ...)` | Teacher queries vs teacher corpus. |
| `evaluate_dir(path, data_path, emb_dir, dim, max_query_len, device, out_dir=None, name=None, ...)` | Any student → val/test metrics + mean cos to teacher queries → `metrics.json` in `out_dir` (default `path`), keyed by `name`. |
| `summarize_runs(dirs, split="test", retention_metric="ndcg@10")` | Table: Recall@100, MRR@10, nDCG@10, retention (student / teacher), mean query cosine to the teacher. |
| `measure_latency(model, tokenizer, texts, device, dim, max_len, batch_sizes, repeats=3)` | Median ms/query per batch size, with `cuda.synchronize`. |

---

## 5. Config keys

### 5.1 `ALIGN_CFG` (`train.ALIGN_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `init` | `"Qwen/Qwen3-Embedding-0.6B"` | Hub id, checkpoint dir, or `"tiny:<seed>"`. |
| `data_path` / `emb_dir` | `cache/data.json` / `cache/emb/teacher` | Built by 00. Targets are the teacher `align` (train) and `val` vectors. |
| `out_dir` | `"runs/aligned"` | Rank 0 saves weights, `cfg.json`, `train_log.jsonl` here. |
| `dim` / `max_query_len` | `256` / `32` | MRL dim (must match the teacher cache) / student query tokens. |
| `loss` / `loss_kwargs` | `"l2"` / `{}` | Any `LOSSES` entry. |
| `lr`, `lr_schedule`, `warmup_ratio`, `min_lr_ratio` | `2e-5`, `"cosine"`, `0.05`, `0.1` | Peak lr, schedule over all optimizer updates. |
| `epochs`, `batch_size`, `accum_steps`, `max_grad_norm` | `1`, `128`, `1`, `1.0` | `batch_size` is **per GPU**; effective batch = `batch_size × GPUs × accum_steps`. |
| `trainable` | `"all_but_embeddings"` | Or `"all"`. |
| `amp` | `True` | fp16 autocast + GradScaler on CUDA; `False` = fp32. |
| `seed`, `val_every`, `save_dtype` | `0`, `200`, `"fp16"` | Val loss + cos to teacher every `val_every` updates and at epoch end. |
| `debug`, `plan_only` | `False`, `False` | 3 steps + shapes + grad check + sync check / print the plan and exit. |
| `max_steps`, `save`, `bench_file` | `None`, `True`, `None` | Used by `benchmark_scaling`. |

### 5.2 `CR_CFG` (`train.CR_DEFAULTS`)
Same keys as §5.1, except:
| Key | Default | Meaning |
|---|---|---|
| `out_dir` | `"runs/cr_only"` | Run B overrides it to `aligned_cr`. |
| `loss` | `"contrastive"` | |
| `loss_kwargs` | `tau=0.05, n_hard_neg=1, same_tower=True, mask_margin=0.1, extra_negs=0` | See `losses.contrastive`. `n_hard_neg` is 0 or 1 (one stored negative per query). |
| `lr` | `1e-5` | |
| `emb_dir` | | Teacher `corpus` vectors are the frozen doc table; `align` vectors give the train queries' targets. |

### 5.3 `TEACHER` (`teacher.TEACHER_DEFAULTS`)
| Key | Default | Meaning |
|---|---|---|
| `model` | `"Qwen/Qwen3-Embedding-4B"` | Debug: `Qwen/Qwen3-Embedding-0.6B` through the same code path. |
| `dim`, `dtype` | `256`, `"fp16"` | MRL dim; teacher weights dtype. |
| `query_prompt` | `QUERY_PROMPT` | Queries only. |
| `doc_max_len` / `query_max_len` | `128` / `64` | Tokens (query length includes the ~20-token prompt). |
| `batch_size` | `64` | Per GPU. |
| `sets` | `("corpus", "align", "val", "test")` | Embedded and merged. |

### 5.4 `DATA` (00 control panel)
`n_rows=210_000`, `core_n=100_000`, `n_val=5_000`, `n_test=5_000`, `n_align=200_000`, `split_seed=1234`,
`use_title=False`. Debug: `12_000 / 5_000 / 500 / 500 / 10_000` (~10k passages).

---

## 6. Tests (`tests/`, CPU, offline, ~25 s)
| Test | Catches |
|---|---|
| `test_pooling_padding_invariant`, `test_tokenize_ends_with_eot` | Pooling depends on padding; pooled token isn't end-of-text. |
| `test_truncate_normalize_unit_norm`, `test_trainable_freezes_embeddings` | Non-unit vectors; wrong trainable set. |
| `test_save_load_roundtrip` | Checkpoints that don't reload identically; missing dir not reported. |
| `test_splits_disjoint_by_query`, `test_corpus_has_positives_and_train_negatives`, `test_alignment_queries_exclude_val_test`, `test_cr_triples` | Leakage between splits, missing positives/negatives, val/test queries in alignment. |
| `test_resolve_cfg_rejects_typos` | Typos in a control panel silently ignored. |
| `test_every_loss_finite_with_grads[*]`, `test_l2_zero_on_equal_inputs` | Any loss in `LOSSES` producing NaN or no gradient. |
| `test_contrastive_masks_same_document`, `test_contrastive_masks_by_margin` | False-negative masking (hand-computed toy cases). |
| `test_retrieve_topk_matches_numpy`, `test_compute_metrics_by_hand` | Retrieval / metric bugs. |
| `test_overfit_one_batch_align`, `test_overfit_one_batch_cr` | The real `train_loop` can't fit one batch (loss must drop > 70%). |
| `test_sharded_embedding_matches_single_process` | 2-process sharded teacher embedding ≠ single process. |
| `test_end_to_end_tiny` | 00 → 01 → 02 → summary logic on 2 gloo processes; debug runs assert DDP weights are identical across ranks after `align_main` and `cr_main`. |
| `test_cr_fails_fast_without_aligned_dir`, `test_benchmark_scaling_runs` | Run B without `aligned/`; scaling benchmark. |
| `test_notebook_clean_and_compiles[*]` | Notebooks invalid, carrying outputs, or with syntax errors. |

Adding a loss, schedule or metric: add it to its registry and to the parametrized test.

## 7. Workflow
1. Edit locally → `.venv/bin/python -m pytest -q` → (notebook cells: `scripts/make_notebooks.py`, then
   `scripts/run_notebooks_tiny.py`) → commit → push.
2. On Kaggle: re-run the setup cell, run with `DEBUG = True`, then `DEBUG = False`.
3. After 01 and 02, re-run the last cell of 00 for the comparison table.

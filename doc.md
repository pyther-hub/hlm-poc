# hlm-poc: aligning a small query encoder to a frozen teacher document index

A frozen **teacher** (Qwen3-Embedding-4B) embeds the document corpus once. Later, a small **student** query
encoder (Qwen3-Embedding-0.6B) will be trained to retrieve against that frozen index, so documents never need
re-embedding when the query side changes.

**Current scope: phase 1 only.** Build the MS MARCO data, embed it with the teacher on 2 GPUs, and evaluate
retrieval on val/test against that index. No fine-tuning yet. Experiments run on Kaggle (GPU T4 x2). This repo holds
the flat, tested functions the notebooks import: plain functions, no classes, no global state, no config system.

---

## 0. Design

### 0.1 Modules
```
       notebooks/phase_1_teacher.ipynb (4 steps)        notebooks/phase_1_eval.ipynb (5 steps)
              │                 │                                  │
       ┌──────▼───┐       ┌─────▼──────┐                  ┌────────▼───────┐
       │ data.py  │       │ teacher.py │  embed_main,     │ evaluation.py  │  load_index (checks), teacher_ceiling,
       │ rows,    │       │            │  merge_shards,   │                │  evaluate_model, METRICS, summarize,
       │ splits,  │       └─────┬──────┘  fp16_sanity,    └────────┬───────┘  measure_latency
       │ corpus   │             │         time_embedding           │
       └────┬─────┘       ┌─────▼──────────────────────────────────▼─┐
            └────────────▶│ model.py  load_encoder · tokenize · encode│
                          └─────┬────────────────────────────────────┘
                          ┌─────▼─────┐
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
```

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
| `load_encoder(name, device, dtype=fp32, device_map=None)` | `name` = hub id or local dir. Returns a frozen, eval-mode `(model, tokenizer)`. `AutoModel` with sdpa. A missing local dir raises `FileNotFoundError`. `device_map="auto"` spreads a model over GPUs. |
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
| `test_expected_notebooks`, `test_notebook_clean_and_compiles` | Stray notebooks, stored outputs, syntax errors. |

## 6. Workflow
1. Edit locally → `.venv/bin/python -m pytest -q` → commit → push.
2. On Kaggle: re-run the setup cell, run with `DEBUG = True`, then `DEBUG = False`.
3. After the teacher index exists: run `phase_1_eval` with the same `DEBUG` value.

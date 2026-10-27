# hlm-poc

Proof of concept: a small student query encoder (Qwen3-Embedding-0.6B) will be trained to retrieve against a frozen
teacher document index (Qwen3-Embedding-4B, d=256). **Current scope: phase 1 only, teacher embeddings. No
fine-tuning yet.** The experiment runs in `notebooks/phase_1_teacher.ipynb` on Kaggle (GPU T4 x2). This repo only
holds clean, tested functions it imports. Same conventions as tiny-reasoner: flat files, plain functions, no classes,
no global state, no config system. Keep the code raw, simple and easy to debug. Pruning is out of scope.

## Layout
```
model.py       # load_encoder (hub id | local dir), tokenize, pool_last_token, truncate_normalize, embed, encode
data.py        # MS MARCO rows, by-query splits, corpus, alignment queries, JSON/emb I/O, resolve_cfg
teacher.py     # TEACHER_DEFAULTS, embed_main (sharded, resumable), merge_shards, fp16_sanity, time_embedding
ddp.py         # copied verbatim from tiny-reasoner; keep the API identical
notebooks/     # phase_1_teacher.ipynb, edited by hand, committed without outputs
tests/         # CPU, offline; the tiny random Qwen3 + GPT-2 tokenizer live ONLY here (conftest.py)
doc.md         # code docs: design + API reference + cfg tables. Update it when a public function or cfg key changes.
README.md      # the user's own notes on experiments and the project. Do not edit unless asked.
```

## Environment
- Always use the project venv: `.venv/bin/python`, `.venv/bin/python -m pytest`. Never system python, never another env.
- Install with `uv pip install --python .venv/bin/python <pkg>` and add it to `requirements.txt`.
- Local machine is a Mac (CPU). Never load the real 0.6B/4B models locally; tests use the tiny model fixture.
  Downloading a tokenizer alone to check its behaviour is fine.

## After EVERY edit
```
.venv/bin/python -m pytest -q                    # ~10 s, must stay < 30 s
```
Do not report a change as done unless this passes.

## Rules
- Modes: `DEBUG=True` (small data, same models) or full. No tiny/test mode in the library or notebook.
- Every result-affecting knob lives in the notebook control panel and in `TEACHER_DEFAULTS`.
  Nothing result-affecting hard-coded in a module.
- DDP: workers live in .py files with signature `(rank, world_size, cfg)`; print/save only on rank 0; every rank hits
  the same collectives.
- New module or model change → test in `tests/`.
- Docs and notebooks use `from model import ...`, never `import model`.
- Data provenance: `negative_passages` are NOT verified to be BM25. Don't claim they are.

## Git
Commit small, focused changes after tests pass. Notebooks are committed without outputs (a test enforces it).

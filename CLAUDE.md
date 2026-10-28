# hlm-poc

Proof of concept: a small student query encoder (Qwen3-Embedding-0.6B) is trained to retrieve against a frozen
teacher document index (Qwen3-Embedding-4B, d=256). **Current scope: phase 1 (teacher embeddings + their evaluation),
phase 2 (alignment fine-tuning: LoRA r=16 + linear head, adapter-only checkpoints) and phase 3 (contrastive
refinement of that adapter against the frozen teacher doc vectors).** Experiments run in
`notebooks/phase_1_teacher.ipynb`, `notebooks/phase_1_eval.ipynb`, `notebooks/phase_2_align.ipynb` and
`notebooks/phase_3_refine.ipynb` on Kaggle (GPU T4 x2). This repo only
holds clean, tested functions it imports. Same conventions as tiny-reasoner: flat files, plain functions, no classes,
no global state, no config system. Keep the code raw, simple and easy to debug. Pruning is out of scope.

## Working style (push back, don't just comply)
Act as a skeptical senior engineer, not an order-taker.
- Before changing anything, restate the goal in one sentence. If the request is ambiguous in a way that would change
  the result, ask first (use the question tool), at most 3 questions per round. Don't ask about trivia.
- If a request has a problem (wrong assumption, bug risk, conflict with these rules or the tests, needless
  complexity, a simpler alternative), STOP before editing. State the problem, the evidence and your recommendation,
  then wait for my decision. Never work around it silently.
- If a request would break a rule in this file, say which rule and ask whether to proceed.
- If I'm wrong, say so directly and explain why. Don't agree to be agreeable.
- No silent scope expansion. For multi-file or design changes, propose a short plan (files to touch, why) and wait for
  approval before editing.
- Prefer deleting code over adding it. No new abstractions, flags or config unless asked.
- Verify instead of asserting: run the tests or read the code. Say what you did not verify.

## Layout notes
- `ddp.py` is copied verbatim from tiny-reasoner; keep the API identical.
- Notebooks are edited by hand and committed without outputs.
- The tiny random Qwen3 + GPT-2 tokenizer live ONLY in `tests/conftest.py`.
- `doc.md` holds the code docs (design, API, cfg tables); update it when a public function or cfg key changes.
- `README.md` is the user's own notes on experiments. Do not edit unless asked.

## Environment
- Always use the project venv: `.venv/bin/python`, `.venv/bin/python -m pytest`. Never system python, never another env.
- Install with `uv pip install --python .venv/bin/python <pkg>` and add it to `requirements.txt`
  (cleanup tools go in `requirements-dev.txt`, see Cleanup).
- Local machine is a Mac (CPU). Never load the real 0.6B/4B models locally; tests use the tiny model fixture.
  Downloading a tokenizer alone to check its behaviour is fine.

## After EVERY edit
```
.venv/bin/python -m pytest -q                    # ~10 s, must stay < 30 s
```
Do not report a change as done unless this passes.

## Rules
- Modes: `DEBUG=True` (small data, same models) or full. No tiny/test mode in the library or notebook.
- Every result-affecting knob lives in the notebook control panel and in `TEACHER_DEFAULTS` / `ALIGN_DEFAULTS` /
  `REFINE_DEFAULTS`.
  Nothing result-affecting hard-coded in a module.
- DDP: workers live in .py files with signature `(rank, world_size, cfg)`; print/save only on rank 0; every rank hits
  the same collectives.
- New module or model change → test in `tests/`.
- Docs and notebooks use `from model import ...`, never `import model`.
- Data provenance: `negative_passages` are NOT verified to be BM25. Don't claim they are.

## Cleanup (only when asked)
Behaviour-preserving only. Never change tests in the same step as source, and never weaken a test to make cleanup pass.
1. Start from a clean tree with tests green; commit first.
2. Report before fixing, using the venv (`requirements-dev.txt`): `ruff check .`, `vulture . --min-confidence 80`,
   `radon cc -s -n C .`. Show me the findings and what you propose to change.
3. Apply only what I approve. Use `/simplify` for reuse, quality and efficiency, with a focus I give you.
4. Re-run the tests, then show `/diff` before committing.
- Vulture false positives: functions used only from notebooks, passed by name to `launch`, or looked up in registries
  (LOSSES, schedules) look dead but are not. Before deleting any public function, grep `notebooks/` and `doc.md`
  and ask me.

## Git
Commit small, focused changes after tests pass. Notebooks are committed without outputs (a test enforces it).
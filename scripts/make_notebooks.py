"""Generate notebooks/00_baseline.ipynb, 01_align_only.ipynb, 02_cr_only.ipynb (no outputs).
Edit the cells here, then: .venv/bin/python scripts/make_notebooks.py"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

SETUP = '''# setup: clone/pull the repo on Kaggle, put it on sys.path, install what's missing (re-run to pull latest code)
import os, sys, importlib.util
REPO_URL = "https://github.com/pyther-hub/hlm-poc.git"
ON_KAGGLE = os.path.exists("/kaggle/working")
REPO = "/kaggle/working/hlm-poc" if ON_KAGGLE else os.path.abspath("..")
if ON_KAGGLE:
    if not os.path.exists(REPO):
        !git clone -q {REPO_URL} {REPO}
    else:
        !git -C {REPO} pull -q
sys.path.insert(0, REPO)

import transformers
from packaging.version import Version
if Version(transformers.__version__) < Version("4.56"):  # Qwen3 + the dtype= kwarg
    !pip install -q -U "transformers>=4.56"
    print("transformers upgraded: restart the kernel, then re-run this cell")
for pkg in ("datasets", "pyarrow", "accelerate"):
    if importlib.util.find_spec(pkg) is None:
        !pip install -q {pkg}

%load_ext autoreload
%autoreload 2'''

COMMON_PANEL = '''import torch

DEBUG = True    # first pass: ~10k passages, Qwen3-Embedding-0.6B as debug teacher, 3 steps per loop, shape printing
TINY = False    # local CPU smoke run: tiny random models + fake rows, no downloads
NPROCS = torch.cuda.device_count() or 2   # DDP processes (2 CPU processes with gloo when there is no GPU)

# --- paths: each mode gets its own cache + runs dirs, so a debug pass never overwrites a real run
ROOT = "/kaggle/working" if ON_KAGGLE else REPO
MODE = "tiny" if TINY else "debug" if DEBUG else "full"
PATHS = dict(cache_dir=f"{ROOT}/cache/{MODE}", runs_dir=f"{ROOT}/runs/{MODE}")
PATHS.update(data_path=f"{PATHS['cache_dir']}/data.json",       # built by 00, cell 1
             emb_dir=f"{PATHS['cache_dir']}/emb/teacher")         # built by 00, cell 2

# --- models
MODELS = dict(
    teacher="tiny:1" if TINY else "Qwen/Qwen3-Embedding-0.6B" if DEBUG else "Qwen/Qwen3-Embedding-4B",
    student="tiny:0" if TINY else "Qwen/Qwen3-Embedding-0.6B",
)
DIM = 32 if TINY else 256   # MRL dim shared by teacher index and student (tiny model has hidden 64)'''


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


def saved_files_cell(*dir_exprs):
    dirs = ", ".join(dir_exprs)
    return code(f'''# what was saved
for out in ({dirs},):
    print(out)
    for f in sorted(os.listdir(out)):
        print(f"  {{os.path.getsize(os.path.join(out, f)) / 1e6:10.2f}} MB  {{f}}")
    with open(os.path.join(out, "train_log.jsonl")) as fh:
        print("  last log:", fh.readlines()[-1].strip())
print("Next: re-run the last cell of 00_baseline.ipynb to evaluate these checkpoints.")''')


def baseline_nb():
    return [
        md("# 00 · Baselines and evaluation (no training)\n"
           "Builds the data, embeds it once with the frozen teacher, reports the teacher ceiling and the untrained "
           "student, and holds the reusable evaluation. **Re-run the last cell after every training notebook.**\n\n"
           "Run with `DEBUG = True` first."),
        code(SETUP),
        md("## Control panel"),
        code(COMMON_PANEL + '''

# --- data (MS MARCO, Tevatron/msmarco-passage train split)
DATA = dict(n_rows=210_000,     # first rows of the split (streamed, cached as parquet)
            core_n=100_000,     # seeded sample of queries -> train / val / test (split by query)
            n_val=5_000, n_test=5_000,
            n_align=200_000,    # alignment queries = train + other queries, never val/test
            split_seed=1234,    # independent of the training seed
            use_title=False)    # passage text = "title. text" when True
if DEBUG:
    DATA.update(n_rows=12_000, core_n=5_000, n_val=500, n_test=500, n_align=10_000)   # ~10k passages
if TINY:
    DATA.update(n_rows=60, core_n=40, n_val=6, n_test=6, n_align=50)

# --- teacher embedding (data-parallel, one shard per GPU, resumable)
TEACHER = dict(model=MODELS["teacher"], dim=DIM, dtype="fp32" if TINY else "fp16",
               doc_max_len=128, query_max_len=64, batch_size=64,
               data_path=PATHS["data_path"], emb_dir=PATHS["emb_dir"])
CHECKS = dict(sanity_n=1000,    # passages for the fp16-vs-fp32 check
              timing_n=2000,    # passages timed for the projection
              budget_h=3.0)     # over budget -> halve corpus + alignment queries

# --- evaluation
EVAL = dict(dim=DIM, max_query_len=32, k=100, batch_size=256, debug=TINY)
RUN_NATIVE = False              # optional reference row: the student in its own embedding space
RUN_LATENCY = False             # optional: query-encoding latency of the student
LATENCY = dict(n_queries=4000, batch_sizes=(1, 4, 8, 16, 32, 64), repeats=3)
BASELINES = f"{PATHS['runs_dir']}/baselines"
EVAL_DIRS = [BASELINES] + [f"{PATHS['runs_dir']}/{d}" for d in ("aligned", "cr_only", "aligned_cr")]'''),
        md("## 1 · Build data\nRows → by-query splits → corpus (positives + one stored negative per core query, "
           "deduplicated) → alignment queries → CR triples. Cached to `data.json`."),
        code('''from data import load_rows, tiny_rows, build_data, save_json, load_json

def make_data(data_cfg):
    rows = tiny_rows(data_cfg["n_rows"]) if TINY else load_rows(data_cfg["n_rows"], PATHS["cache_dir"])
    d = build_data(rows, **{k: v for k, v in data_cfg.items() if k != "n_rows"})
    save_json(d, PATHS["data_path"])
    return rows, d

if os.path.exists(PATHS["data_path"]):
    data = load_json(PATHS["data_path"])
    print("[data] cached:", {k: len(v["ids"]) for k, v in data.items()})
else:
    rows, data = make_data(DATA)
    print("one row:", {k: (v if k == "query" else str(v)[:120]) for k, v in rows[0].items()})'''),
        md("## 2 · Teacher embeddings\nfp16 sanity check and a timing projection first, then the full run on all GPUs. "
           "Corpus gets no prefix; queries get the instruction prompt. Shards that already exist are skipped."),
        code('''from ddp import launch
from model import get_device
from teacher import embed_main, fp16_sanity, time_embedding, halve_sizes
from train import clean_unused_gpu_memory

device = get_device()
if not os.path.exists(f"{PATHS['emb_dir']}/corpus.npy"):
    sanity = fp16_sanity(TEACHER["model"], data["corpus"]["texts"][:CHECKS["sanity_n"]], DIM, TEACHER["doc_max_len"],
                         device, device_map_fp32="auto" if torch.cuda.device_count() > 1 else None)
    assert sanity["nan_rows"] == 0, "fp16 teacher produced NaNs"
    timing = time_embedding(TEACHER, data, device, world_size=NPROCS, n_sample=CHECKS["timing_n"],
                            budget_h=CHECKS["budget_h"])
    if not timing["go"]:
        DATA = halve_sizes(DATA)
        print("[go/no-go] over budget, rebuilding with", DATA)
        rows, data = make_data(DATA)
    clean_unused_gpu_memory()
launch(embed_main, TEACHER, nprocs=NPROCS)'''),
        md("## 3 · Teacher ceiling\nTeacher queries against the teacher corpus index: the best the student can reach."),
        code('''from evaluation import teacher_ceiling, evaluate_dir, summarize_runs

EV = dict(data_path=PATHS["data_path"], emb_dir=PATHS["emb_dir"], dim=EVAL["dim"], max_query_len=EVAL["max_query_len"],
          device=device, k=EVAL["k"], batch_size=EVAL["batch_size"], debug=EVAL["debug"])
teacher_ceiling(PATHS["data_path"], PATHS["emb_dir"], BASELINES, k=EVAL["k"], device=device, debug=EVAL["debug"])'''),
        md("## 4 · Untrained student against the teacher index\n"
           "Expected to score **near zero**: the student's embedding space is not the teacher's, so its query vectors "
           "do not line up with the teacher's document vectors even though both models are good retrievers on their own. "
           "The mean cos(student query, teacher query) on val shows the gap that alignment has to close."),
        code('''untrained = evaluate_dir(MODELS["student"], out_dir=BASELINES, name="untrained_student", **EV)
print("mean cos(student q, teacher q) on val:", round(untrained["val"]["cos_to_teacher"], 4))'''),
        md("## 5 · Optional: student in its own native space\nEmbeds the corpus with the student (cached separately) "
           "and uses the student's own instruction prompt for queries, as intended by its model card. "
           "A reference row only; `cos_to_teacher` is meaningless for it."),
        code('''if RUN_NATIVE:
    import numpy as np
    from model import load_encoder, encode
    from teacher import QUERY_PROMPT
    native_path = f"{PATHS['cache_dir']}/emb/student_native/corpus.npy"
    if not os.path.exists(native_path):
        m, tok = load_encoder(MODELS["student"], device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        emb = encode(m, tok, data["corpus"]["texts"], device, DIM, TEACHER["doc_max_len"], progress=True)
        os.makedirs(os.path.dirname(native_path), exist_ok=True)
        np.save(native_path, emb.numpy())
        del m; clean_unused_gpu_memory()
    evaluate_dir(MODELS["student"], out_dir=BASELINES, name="student_native", doc_emb_path=native_path,
                 query_prefix=QUERY_PROMPT, **EV)'''),
        md("## Optional · Query-encoding latency (student)"),
        code('''if RUN_LATENCY:
    from evaluation import measure_latency
    from model import load_encoder
    m, tok = load_encoder(MODELS["student"], device)
    measure_latency(m, tok, data["test"]["texts"][:LATENCY["n_queries"]], device, DIM, EVAL["max_query_len"],
                    LATENCY["batch_sizes"], LATENCY["repeats"])
    del m; clean_unused_gpu_memory()'''),
        md("## 6 · Evaluate checkpoints + summary\nEvaluates every trained dir in `EVAL_DIRS` that exists "
           "(writes `metrics.json` into it), then prints one table, teacher row first. **Re-run after each training notebook.**"),
        code('''for d in EVAL_DIRS[1:]:
    if os.path.exists(f"{d}/config.json"):
        evaluate_dir(d, **EV)
    else:
        print("[eval] not trained yet:", d)
clean_unused_gpu_memory()
rows = summarize_runs(EVAL_DIRS, split="test")'''),
    ]


def align_nb():
    return [
        md("# 01 · Alignment (stage A)\nTrains the student so its query vectors match the teacher's cached query vectors "
           "(no teacher forward pass). Output: `aligned/` (weights, cfg.json, train_log.jsonl). "
           "Retrieval is evaluated in 00, not here.\n\nRun 00 cells 1–2 first. Run with `DEBUG = True` first."),
        code(SETUP),
        md("## Control panel"),
        code(COMMON_PANEL + '''

ALIGN_CFG = dict(
    # model
    init=MODELS["student"],           # hub id or checkpoint dir
    dim=DIM, max_query_len=32,
    trainable="all_but_embeddings",   # all | all_but_embeddings (token embeddings frozen -> ~0.44B trainable)
    # loss (see losses.py)
    loss="l2", loss_kwargs={},        # l2 | cosine | hybrid(terms={...})
    # optimisation
    lr=2e-5, lr_schedule="cosine",    # cosine | constant
    warmup_ratio=0.05, min_lr_ratio=0.1,
    epochs=1, batch_size=128,         # batch_size is per GPU
    accum_steps=1, max_grad_norm=1.0,
    amp=True,                         # fp16 autocast + GradScaler; False = fp32 (use if you see NaNs)
    # run
    seed=0, val_every=200, save_dtype="fp16",
    debug=DEBUG,
    # io
    data_path=PATHS["data_path"], emb_dir=PATHS["emb_dir"], out_dir=f"{PATHS['runs_dir']}/aligned",
)
if TINY:
    ALIGN_CFG.update(batch_size=4, max_query_len=16, amp=False)
RUN_SCALING = True
SCALING_STEPS = 3 if (DEBUG or TINY) else 20'''),
        md("## Plan\nResolved config, effective batch, optimizer updates, trainable/total params, first-batch shapes. Trains nothing."),
        code('''from ddp import launch
from train import align_main, benchmark_scaling

launch(align_main, {**ALIGN_CFG, "plan_only": True}, nprocs=NPROCS)'''),
        md("## Scaling: 1 vs 2 processes\nSamples/s and scaling efficiency (warm-up steps discarded, no checkpoints)."),
        code('''if RUN_SCALING and NPROCS > 1:
    benchmark_scaling(align_main, ALIGN_CFG, steps=SCALING_STEPS, nprocs_list=(1, NPROCS))'''),
        md("## Debug run\n3 steps per GPU, tensor shapes, grad check, weights-in-sync check. Saves to a separate dir."),
        code('''launch(align_main, {**ALIGN_CFG, "debug": True, "out_dir": f"{PATHS['runs_dir']}/debug_aligned"}, nprocs=NPROCS)'''),
        md("## Real run"),
        code('''launch(align_main, ALIGN_CFG, nprocs=NPROCS)'''),
        saved_files_cell('ALIGN_CFG["out_dir"]'),
    ]


def cr_nb():
    return [
        md("# 02 · Contrastive refinement (stage C), two runs\n"
           "CR against the cached teacher document vectors (a frozen lookup table, no document encoder).\n"
           "- **Run A**: from the original pretrained student → `cr_only/`\n"
           "- **Run B**: from the aligned checkpoint → `aligned_cr/` (needs 01 first)\n\n"
           "The two runs differ only in `init` and `out_dir`. Run with `DEBUG = True` first."),
        code(SETUP),
        md("## Control panel"),
        code(COMMON_PANEL + '''

CR_CFG = dict(
    # model
    dim=DIM, max_query_len=32,
    trainable="all_but_embeddings",
    # loss (see losses.py)
    loss="contrastive",
    loss_kwargs=dict(tau=0.05,          # temperature
                     n_hard_neg=1,      # 0 or 1: the corpus keeps one stored negative per query
                     same_tower=True,   # other queries in the batch as negatives
                     mask_margin=0.1,   # mask negatives scoring > positive + margin (likely false negatives)
                     extra_negs=0),     # extra random negatives drawn from the corpus table per step
    # optimisation
    lr=1e-5, lr_schedule="cosine", warmup_ratio=0.05, min_lr_ratio=0.1,
    epochs=1, batch_size=128, accum_steps=1, max_grad_norm=1.0,
    amp=True,
    # run
    seed=0, val_every=200, save_dtype="fp16",
    debug=DEBUG,
    # io
    data_path=PATHS["data_path"], emb_dir=PATHS["emb_dir"],
)
if TINY:
    CR_CFG.update(batch_size=4, max_query_len=16, amp=False)

RUN_A = dict(init=MODELS["student"],                  out_dir=f"{PATHS['runs_dir']}/cr_only")
RUN_B = dict(init=f"{PATHS['runs_dir']}/aligned",     out_dir=f"{PATHS['runs_dir']}/aligned_cr")'''),
        md("## Plan (Run A)"),
        code('''from ddp import launch
from train import cr_main

launch(cr_main, {**CR_CFG, **RUN_A, "plan_only": True}, nprocs=NPROCS)'''),
        md("## Debug run\n3 steps per GPU, tensor shapes, grad check, weights-in-sync check. Saves to a separate dir."),
        code('''launch(cr_main, {**CR_CFG, **RUN_A, "debug": True, "out_dir": f"{PATHS['runs_dir']}/debug_cr"}, nprocs=NPROCS)'''),
        md("## Run A: CR from the pretrained student"),
        code('''launch(cr_main, {**CR_CFG, **RUN_A}, nprocs=NPROCS)'''),
        md("## Run B: CR from the aligned checkpoint"),
        code('''if not os.path.exists(f"{RUN_B['init']}/config.json"):
    raise FileNotFoundError(f"{RUN_B['init']} not found: run 01_align_only.ipynb (real run) first")
launch(cr_main, {**CR_CFG, **RUN_B}, nprocs=NPROCS)'''),
        saved_files_cell('RUN_A["out_dir"]', 'RUN_B["out_dir"]'),
    ]


def write(name, cells):
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbformat.validate(nb)
    path = os.path.join(OUT, name)
    nbformat.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write("00_baseline.ipynb", baseline_nb())
    write("01_align_only.ipynb", align_nb())
    write("02_cr_only.ipynb", cr_nb())

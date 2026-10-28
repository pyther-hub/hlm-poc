"""Experiments on top of the phases: one plain function per experiment, called from notebooks/experiments.ipynb.
Every arm is a run dir <exp_dir>/<arm>/ (adapter checkpoints + log.jsonl, as in align / refine / prune) and a row in
<exp_dir>/metrics.json; plots go to <exp_dir>/plots/. An arm whose final/ exists is not retrained and a row already
in metrics.json is not re-evaluated, so re-running an experiment (e.g. in a new Kaggle session) resumes it.
The teacher is frozen and its cached vectors are reused; nothing here re-embeds the corpus.
  exp_losses   alignment loss: L2 vs KL (KaLM-style)
  exp_oneshot  one-shot contrastive training straight from the base student vs align -> CR
  exp_prune    progressive vs direct pruning
  exp_cr       before vs after contrastive refinement, full and pruned students
  exp_latency  query-encoding latency / throughput per model + the cos(q, d+) shift across phases"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import align
from align import align_main, read_log
from data import load_json, save_json
from ddp import launch
from evaluation import evaluate_model, measure_latency, save_metrics, teacher_ceiling
from model import attach_adapter, encode, load_encoder, save_adapter
from prune import prune_main
from refine import refine_main
from teacher import free_gpu_memory


# ---------- experimental loss (only for exp_losses; L2 stays the default everywhere) ----------

def kl_loss(student, teacher, tau=0.05):
    """KaLM-style distillation: KL(p_T || p_S), where p_X(i, .) = softmax_j(x_i . t_j / tau) over the batch's teacher
    query vectors t_j."""
    log_p_student = F.log_softmax(student @ teacher.T / tau, -1)
    p_teacher = F.softmax(teacher @ teacher.T / tau, -1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean")


def exp_align_main(rank, world_size, cfg):
    """align_main with kl_loss available as loss="kl". Registered here, in the spawned worker: a registration made in
    the notebook process doesn't reach the workers."""
    align.LOSSES["kl"] = kl_loss
    align_main(rank, world_size, cfg)


# ---------- shared ----------

def train_once(worker, cfg, nprocs):
    """launch(worker, cfg) unless cfg["out_dir"]/final exists. Returns the final dir."""
    final = os.path.join(cfg["out_dir"], "final")
    if not os.path.exists(final):
        launch(worker, cfg, nprocs=nprocs)
    return final


def evaluate_runs(runs, data, index, exp_dir, ev):
    """runs: {row name: model dir}; index = evaluation.load_index(...) = (teacher docs, {split: teacher queries}).
    Evaluates the rows not yet in <exp_dir>/metrics.json (ev = evaluate_model kwargs: splits, dim, max_query_len,
    query_prefix, device, dtype), plus the teacher ceiling row. Returns all metrics."""
    docs, teacher_q = index
    path = os.path.join(exp_dir, "metrics.json")
    done = load_json(path) if os.path.exists(path) else {}
    if "teacher" not in done:
        save_metrics(exp_dir, teacher_ceiling(data, docs, teacher_q, device=ev["device"]))
    for name, model_dir in runs.items():
        if name not in done:
            save_metrics(exp_dir, {name: evaluate_model(model_dir, data, docs, **ev)})
    return load_json(path)


def save_fig(fig, exp_dir, name):
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    fig.savefig(os.path.join(exp_dir, "plots", f"{name}.png"), dpi=120, bbox_inches="tight")
    return fig


def plot_curves(run_dirs, keys=("train_loss", "val_ndcg@10"), title=""):
    """One panel per log key, one line per run: {label: run dir with log.jsonl}."""
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for label, d in run_dirs.items():
            rows = [r for r in read_log(d) if key in r]
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], marker="o" if key.startswith("val") else None,
                    markersize=3, label=label)
        ax.set(xlabel="step", ylabel=key, title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bars(metrics, names, split="test", keys=("recall@100", "ndcg@10", "mrr@10"), title=""):
    """Grouped bars: one group per metric, one bar per row name, value on top."""
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(keys) * len(names) / 2, 3.8))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [metrics[name][split][k] for k in keys]
        bars = ax.bar(np.arange(len(keys)) + i * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(np.arange(len(keys)) + width * (len(names) - 1) / 2, keys)
    ax.set(ylim=(0, 1.05), title=title or f"split={split}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def print_rows(metrics, names, split="test", keys=("recall@100", "recall@10", "ndcg@10", "mrr@10")):
    """Table of the given rows, with retention = ndcg@10 / teacher ndcg@10."""
    base = metrics["teacher"][split]["ndcg@10"]
    print(f"split={split}\n{'run':<22}" + "".join(f"{k:>11}" for k in keys) + f"{'retention':>11}")
    for name in ["teacher"] + [n for n in names if n != "teacher"]:
        m = metrics[name][split]
        print(f"{name:<22}" + "".join(f"{m[k]:>11.4f}" for k in keys) + f"{m['ndcg@10'] / base:>11.4f}")


# ---------- experiments ----------

def exp_losses(align_cfg, losses, data, index, exp_dir, ev, nprocs):
    """#3 alignment loss. losses: {arm: (loss name, loss_kwargs)}, e.g. {"align_l2": ("l2", {}),
    "align_kl": ("kl", {"tau": 0.05})}. Each arm = phase 2 from scratch with that loss, everything else align_cfg."""
    runs = {}
    for arm, (loss, kwargs) in losses.items():
        cfg = {**align_cfg, "loss": loss, "loss_kwargs": kwargs, "out_dir": os.path.join(exp_dir, arm)}
        runs[arm] = train_once(exp_align_main, cfg, nprocs)
    metrics = evaluate_runs(runs, data, index, exp_dir, ev)
    dirs = {arm: os.path.join(exp_dir, arm) for arm in losses}
    figs = [save_fig(plot_curves(dirs, ("train_loss", "val_loss", "val_ndcg@10"), "alignment loss"), exp_dir,
                     "losses_curves"),
            save_fig(plot_bars(metrics, list(losses), title="alignment loss (test)"), exp_dir, "losses_bars")]
    print("train_loss is each arm's own loss; val_loss is L2 for every arm (comparable)")
    print_rows(metrics, list(losses))
    return dict(metrics=metrics, figs=figs)


def exp_cr(refine_cfg, students, data, index, exp_dir, ev, nprocs):
    """#2 before vs after CR. students: {name: adapter dir}, e.g. the phase-2 adapter and the pruned 4-layer ones.
    Each gets refine_cfg (row "<name>_cr"); a pruned (full) student trains every weight, a LoRA one its adapter."""
    runs = dict(students)
    for name, init in students.items():
        cfg = {**refine_cfg, "init_adapter": init, "out_dir": os.path.join(exp_dir, f"{name}_cr")}
        runs[f"{name}_cr"] = train_once(refine_main, cfg, nprocs)
    metrics = evaluate_runs(runs, data, index, exp_dir, ev)
    for name in students:
        before, after = metrics[name]["test"], metrics[f"{name}_cr"]["test"]
        print(f"{name:<14} CR gain: recall@100 {after['recall@100'] - before['recall@100']:+.4f}  "
              f"ndcg@10 {after['ndcg@10'] - before['ndcg@10']:+.4f}")
    names = [n for s in students for n in (s, f"{s}_cr")]
    figs = [save_fig(plot_bars(metrics, names, title="before vs after CR (test)"), exp_dir, "cr_bars"),
            save_fig(plot_curves({f"{s}_cr": os.path.join(exp_dir, f"{s}_cr") for s in students},
                                 ("cr", "val_ndcg@10"), "CR training"), exp_dir, "cr_curves")]
    print_rows(metrics, names)
    return dict(metrics=metrics, figs=figs)


def exp_oneshot(align_cfg, refine_cfg, aligned, data, index, exp_dir, ev, nprocs):
    """#1 proxy. "oneshot" = a fresh adapter on the base student trained only with L_CR (l2_weight=0) against the frozen
    teacher docs, for about as many steps as align + CR together. Compared with "aligned" (phase-2 adapter dir) and
    "aligned_cr" (shared with exp_cr when run there first). Not the paper's baseline: the doc tower isn't trained."""
    init = os.path.join(exp_dir, "oneshot_init")
    if not os.path.exists(init):
        model, _ = load_encoder(align_cfg["model"], torch.device("cpu"))
        torch.manual_seed(align_cfg["seed"])
        attach_adapter(model, align_cfg["lora_rank"], align_cfg["lora_alpha"], align_cfg["lora_dropout"],
                       align_cfg["lora_targets"], align_cfg["dim"])
        save_adapter(model, init, dict(base=align_cfg["model"], rank=align_cfg["lora_rank"],
                                       alpha=align_cfg["lora_alpha"], dropout=align_cfg["lora_dropout"],
                                       targets=list(align_cfg["lora_targets"]), out_dim=align_cfg["dim"]))
        del model
    align_steps = len(data["align"]["ids"]) // (align_cfg["batch_size"] * nprocs) * align_cfg["epochs"]
    cr_per_epoch = len(data["train"]["ids"]) // (refine_cfg["batch_size"] * nprocs)
    epochs = max(1, round((align_steps + cr_per_epoch * refine_cfg["epochs"]) / cr_per_epoch))
    print(f"[oneshot] align {align_steps} + CR {cr_per_epoch * refine_cfg['epochs']} steps -> one-shot {epochs} "
          f"epochs = {epochs * cr_per_epoch} steps")
    runs = {"aligned": aligned}
    runs["aligned_cr"] = train_once(refine_main, {**refine_cfg, "init_adapter": aligned,
                                                  "out_dir": os.path.join(exp_dir, "aligned_cr")}, nprocs)
    runs["oneshot"] = train_once(refine_main, {**refine_cfg, "init_adapter": init, "l2_weight": 0.0, "epochs": epochs,
                                               "out_dir": os.path.join(exp_dir, "oneshot")}, nprocs)
    metrics = evaluate_runs(runs, data, index, exp_dir, ev)
    figs = [save_fig(plot_bars(metrics, list(runs), title="one-shot CL vs align -> CR (test)"), exp_dir,
                     "oneshot_bars"),
            save_fig(plot_curves({"oneshot": os.path.join(exp_dir, "oneshot"),
                                  "aligned_cr": os.path.join(exp_dir, "aligned_cr")}, ("cr", "val_ndcg@10"),
                                 "one-shot vs CR after alignment"), exp_dir, "oneshot_curves")]
    print_rows(metrics, list(runs))
    return dict(metrics=metrics, figs=figs)


def exp_prune(prune_cfg, aligned, stages, direct, data, index, exp_dir, ev, nprocs):
    """#5 progressive vs direct pruning, full fine-tune at every stage. Progressive: aligned -> stages[0] -> stages[1]
    ... layers ("prog_l<n>"), each stage cutting prune_cfg["ffn_keep"] of the FFN width. Direct: aligned -> `direct`
    layers in one cut ("direct_l<n>") with the same final FFN width (ffn_keep ** stages used to reach it)."""
    runs, init = {"aligned": aligned}, aligned
    for n in stages:
        cfg = {**prune_cfg, "init": init, "n_layers": n, "out_dir": os.path.join(exp_dir, f"prog_l{n}")}
        init = runs[f"prog_l{n}"] = train_once(prune_main, cfg, nprocs)
    cuts = stages.index(direct) + 1
    cfg = {**prune_cfg, "init": aligned, "n_layers": direct, "ffn_keep": prune_cfg["ffn_keep"] ** cuts,
           "out_dir": os.path.join(exp_dir, f"direct_l{direct}")}
    runs[f"direct_l{direct}"] = train_once(prune_main, cfg, nprocs)
    metrics = evaluate_runs(runs, data, index, exp_dir, ev)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    layers = [28] + list(stages)
    for key, style in (("recall@100", "-o"), ("ndcg@10", "-s")):
        ax.plot(layers, [metrics[n]["test"][key] for n in ["aligned"] + [f"prog_l{s}" for s in stages]], style,
                label=f"progressive {key}")
        ax.plot([direct], [metrics[f"direct_l{direct}"]["test"][key]], "x", markersize=10, mew=2,
                label=f"direct 28->{direct} {key}")
    ax.set(xlabel="decoder layers", ylabel="test", title="progressive vs direct pruning", xscale="log", ylim=(0, 1))
    ax.set_xticks(layers, [str(n) for n in layers])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    figs = [save_fig(fig, exp_dir, "prune_layers"),
            save_fig(plot_curves({n: os.path.join(exp_dir, n) for n in runs if n != "aligned"},
                                 ("train_loss", "val_ndcg@10"), "pruning recovery"), exp_dir, "prune_curves")]
    print_rows(metrics, list(runs))
    return dict(metrics=metrics, figs=figs)


def cos_to_positive(model_dir, texts, pos, docs, device, dim, max_len, prefix=""):
    """Mean cos(q, d+) between a model's query vectors and the teacher vector of each query's first positive."""
    model, tok = load_encoder(model_dir, device, torch.float16 if device.type == "cuda" else torch.float32)
    q = encode(model, tok, texts, device, dim, max_len, prefix, batch_size=256).float().numpy()
    del model
    free_gpu_memory()
    d = docs[[p[0] for p in pos]].astype(np.float32)
    return float((q * d).sum(1).mean())


def exp_latency(models, data, index, exp_dir, device, dim, n_queries, batch_sizes):
    """#7. models: {name: (model dir or hub id, max_len, prefix)}, the first one is the reference for speedups
    (the teacher). Latency / throughput of query encoding at each batch size (fp16 on GPU) on n_queries val queries,
    then mean cos(q, d+) on the val split for every other model; the teacher's comes from its cached val vectors."""
    docs, teacher_q = index
    texts = data["val"]["texts"][:n_queries]
    path = os.path.join(exp_dir, "latency.json")
    lat = load_json(path) if os.path.exists(path) else {}
    for name, (model_dir, max_len, prefix) in models.items():
        if name in lat:
            continue
        model, tok = load_encoder(model_dir, device, torch.float16 if device.type == "cuda" else torch.float32)
        print(f"[latency] {name}  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
        lat[name] = dict(params=sum(p.numel() for p in model.parameters()),
                         rows=measure_latency(model, tok, texts, device, dim, max_len, prefix, batch_sizes))
        del model
        free_gpu_memory()
        save_json(lat, path)
    ref = next(iter(models))
    print(f"\nspeedup vs {ref} (ms/query ratio)\n{'model':<16}{'params':>9}" + "".join(f"{f'bs={b}':>10}"
                                                                                 for b in batch_sizes))
    for name in models:
        sp = [r0["ms_per_query"] / r["ms_per_query"] for r0, r in zip(lat[ref]["rows"], lat[name]["rows"])]
        print(f"{name:<16}{lat[name]['params'] / 1e6:>8.0f}M" + "".join(f"{s:>9.2f}x" for s in sp))

    pos = [p[0] for p in data["val"]["pos"]]
    cos = {"teacher": float((teacher_q["val"].astype(np.float32) * docs[pos].astype(np.float32)).sum(1).mean())}
    for name, (model_dir, max_len, prefix) in list(models.items())[1:]:
        cos[name] = cos_to_positive(model_dir, data["val"]["texts"], data["val"]["pos"], docs, device, dim, max_len,
                                    prefix)
    print("\nmean cos(q, d+) on val: " + "  ".join(f"{k} {v:.3f}" for k, v in cos.items()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in models:
        axes[0].plot(batch_sizes, [r["queries_per_s"] for r in lat[name]["rows"]], "-o", label=name)
    axes[0].set(xscale="log", yscale="log", xlabel="batch size", ylabel="queries / s", title="query encoding throughput")
    axes[0].set_xticks(batch_sizes, [str(b) for b in batch_sizes])
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(list(cos), list(cos.values()))
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[1].set(ylim=(0, 1.05), title="mean cos(q, d+) vs teacher doc vectors (val)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return dict(latency=lat, cos=cos, figs=[save_fig(fig, exp_dir, "latency_cos")])

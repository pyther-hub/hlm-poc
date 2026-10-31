"""DDP helpers. All of them are safe no-ops when no process group is running,
so train/eval code can call them unconditionally."""
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def launch(fn, *args, nprocs=None):
    """Run fn(rank, world_size, *args) in nprocs processes (default: one per GPU).
    fn must be importable from a .py file — functions defined in a notebook cell can't be spawned."""
    nprocs = nprocs or torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["PYTHONUNBUFFERED"] = "1"  # so worker prints reach the notebook promptly
    mp.spawn(fn, args=(nprocs, *args), nprocs=nprocs, join=True)


def setup(rank, world_size, timeout_s=1800):
    """Join the process group. Returns this rank's device. Use a short timeout in debug runs:
    if one rank crashes, the others otherwise wait in all_reduce until it expires."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        backend, device = "nccl", torch.device(f"cuda:{rank}")
    else:
        backend, device = "gloo", torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=timeout_s))
    return device


def cleanup():
    if is_dist():
        dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def reduce_mean(value, device):
    """Average a python float across ranks (for logging losses)."""
    if not is_dist():
        return value
    t = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / get_world_size()).item()


def wrap_ddp(model, device):
    # gradient_as_bucket_view: grads live in the all-reduce buckets instead of a second fp32 copy (same math,
    # saves 4 bytes per trainable param, ~2.4 GB for a 596M full fine-tune). The one change vs tiny-reasoner.
    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None, gradient_as_bucket_view=True)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

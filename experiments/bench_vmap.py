"""
Time vmapped vs. serial, on cifar10 / cnn (width 32):
- HVPs of `runs` vectors from the same gradients (`hvp_from_grads`: `is_grads_batched` vs. a loop)
- `runs` SGD trainings (the forward vmapped over the models vs. one model after the other), plain and compiled
  python -m experiments.bench_vmap --device cuda [--batch-sizes 64,128] [--runs 10]
"""

import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable

import click
import torch
from torch import Tensor

from main import Experiment, ExperimentConfig
from src.optimizers import OptimState


def timed(fn: Callable[[], object], reps: int, device: str) -> tuple[float, float]:
    """(median seconds per call, peak GPU GB) of `fn`, after one warmup call"""
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else lambda: None
    fn()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2], torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0


def bench_hvp(exp: Experiment, runs: int, reps: int) -> dict[str, tuple[float, float]]:
    magic = exp.magic
    X, Y, _ = exp.loader.get_batch(0)
    params = {k: p.to(exp.device).requires_grad_() for k, p in exp.init_state.params.items()}
    grads = torch.autograd.grad(magic.objective.forward(params, X, Y)[0], params, create_graph=True)
    vectors = {k: torch.randn(runs, *p.shape, device=p.device) for k, p in params.items()}

    def hvp(cotangents: list[Tensor], batched: bool):
        return torch.autograd.grad(list(grads.values()), list(params.values()), cotangents, retain_graph=True, is_grads_batched=batched)

    vmapped = lambda: hvp([vectors[k] for k in grads], batched=True)
    serial = lambda: [hvp([vectors[k][i] for k in grads], batched=False) for i in range(runs)]
    return {"vmapped": timed(vmapped, reps, exp.device), "serial": timed(serial, reps, exp.device)}


def train(exp: Experiment, state: OptimState, forward: Callable, weights: Tensor, n_steps: int) -> OptimState:
    """`Magic.train` with a given `forward` (compiled once) and no logging"""
    for t in range(n_steps):
        X, Y, idx = exp.loader.get_batch(t)
        for p in state.params.values():
            p.requires_grad_()
        state = state.step(t, partial(forward, X=X, Y=Y, weights=weights[..., idx]))[0].apply(lambda t: t.detach())
    return state


def bench_train(exp: Experiment, runs: int, n_steps: int, reps: int, compile: bool) -> dict[str, tuple[float, float]]:
    magic = replace(exp.magic, compile=compile)
    state = exp.init_state.apply(lambda t: t.to(exp.device))
    weights = 1 + 0.05 * torch.randn(runs, len(exp.train_ds), device=exp.device)
    forward, _, _ = magic._train_forward(state, weights[0])
    vmapped_forward, stacked, _ = magic._train_forward(state, weights)

    vmapped = lambda: train(exp, stacked, vmapped_forward, weights, n_steps)
    serial = lambda: [train(exp, state, forward, weights[r], n_steps) for r in range(runs)]
    return {"vmapped": timed(vmapped, reps, exp.device), "serial": timed(serial, reps, exp.device)}


@click.command(help=__doc__)
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--batch-sizes", type=str, default="64,128", show_default=True)
@click.option("--runs", type=int, default=10, show_default=True, help="vectors per HVP call / models trained")
@click.option("--steps", type=int, default=20, show_default=True, help="training steps per timed call")
@click.option("--reps", type=int, default=5, show_default=True, help="timed calls (the median is reported)")
def main(device, batch_sizes, runs, steps, reps):
    click.echo(f"device: {torch.cuda.get_device_name() if device.startswith('cuda') else device}, torch {torch.__version__}")
    for batch_size in (int(b) for b in batch_sizes.split(",")):
        cfg = ExperimentConfig(dataset="cifar10", model="cnn", width=32, lr=0.2, batch_size=batch_size)
        exp = Experiment(Path("."), device, cfg=cfg)
        benches = {f"hvp of {runs} vectors (ms per call)": (lambda: bench_hvp(exp, runs, reps), 1.0)}
        for compile in (False, True):
            name = f"train {runs} models{', compiled' if compile else ''} (ms per step of all models)"
            benches[name] = (lambda compile=compile: bench_train(exp, runs, steps, reps, compile), float(steps))
        for name, (bench, calls) in benches.items():
            try:
                results = bench()
            except Exception as e:  # (e.g. no C++ compiler for torch.compile)
                click.echo(f"batch {batch_size}, {name}: failed ({type(e).__name__}: {str(e)[:200]})")
                continue
            (t_vmap, gb_vmap), (t_serial, gb_serial) = results["vmapped"], results["serial"]
            click.echo(f"batch {batch_size}, {name}: vmapped {1e3 * t_vmap / calls:.1f} ({gb_vmap:.2f} GB), "
                       f"serial {1e3 * t_serial / calls:.1f} ({gb_serial:.2f} GB), serial / vmapped = {t_serial / t_vmap:.2f}")


if __name__ == "__main__":
    main()

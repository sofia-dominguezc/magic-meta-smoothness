"""
Time and peak GPU memory of the central flow (`CentralFlowState`) on an experiment's model and full batch, into DIR/profile/TAG.json
with its tables (TAG.md) and plots (TAG_unit.png, TAG_steps.png); TAG defaults to the GPU name (and _eager with --no-compile):
- "unit": each operation alone (the forward and backward, the HVPs of LOBPCG by variant and number of vectors, eager and compiled,
  the curvature backwards by number of cotangents, a pair's third-derivative graph and its backward, the eigenvector term), to
  build a cost model in the batch size and the number of critical eigenvalues k
- "lobpcg" (part of "unit"): LOBPCG from random vectors with each HVP: torch.func forward-over-backward eager, compiled, and
  stateful autograd through the gradients' graph (no forward recomputed, all the vectors in one batched backward)
- "steps": --steps consecutive steps and the vjp of the last one, by phase (the step: LOBPCG, the loss graphs, the pairs, the flow
  update; the vjp: the loss graphs, the backward to the curvatures, the pairs, MINRES, the eigenvector term), for each k of --ks
  with the lr set so that exactly k eigenvalues are critical at the start, without and with the compiled HVPs of LOBPCG (the first
  compiled step includes the compilation)

  python -m experiments.profile_centralflow DIR [--tag TAG] [--section unit|lobpcg|steps|all] [--n-train 1024] [--ks 1,2,4,8] [--steps 1]
                                            [--device cuda] [--devices cuda:0] [--no-compile]
  python -m experiments.profile_centralflow DIR --tag TAG --report OLD.json     # only the tables and plots of an existing OLD.json
--devices: helper devices (the pairs and LOBPCG's HVP chunks are dealt among them and the state's own); the same device
repeated measures the per-pair path on one GPU (the loss graphs' memory is then repeated).
"""

import gc
import json
import re
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import src.second_order as second_order
from main import Experiment
from src.lobpcg import hessian_eigs
from src.objective import hvp_from_forward, hvp_from_grads
from src.magic import Magic
from src.optimizers import LRSchedule
from src.second_order import COMPILED, CentralFlowState, Loss, LossGraph, curvature_grad, curvature_grad_vjp

GB = 2**30


class Meter:
    """Seconds and GPU memory of nested phases: peak, base (allocated at entry) and retained (allocated at exit - base), in GB"""

    def __init__(self, device: str):
        self.cuda = device.startswith("cuda")
        self.stack: list[dict] = []

    def _allocated(self) -> float:
        return torch.cuda.memory_allocated() / GB if self.cuda else 0.0

    def _roll_peak(self):
        """Fold the peak since the last call into the innermost open phase"""
        if self.cuda:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / GB
            torch.cuda.reset_peak_memory_stats()
            if self.stack:
                self.stack[-1]["peak"] = max(self.stack[-1]["peak"], peak)

    @contextmanager
    def phase(self, sink: dict | None = None, name: str = ""):
        """Measures the block into the yielded dict, and accumulates it into `sink[name]`"""
        self._roll_peak()
        frame = {"peak": 0.0, "base": self._allocated()}
        self.stack.append(frame)
        start = time.perf_counter()
        try:
            yield frame
        finally:
            self._roll_peak()
            frame["seconds"] = time.perf_counter() - start
            self.stack.pop()
            frame["retained"] = self._allocated() - frame["base"]
            frame["extra"] = frame["peak"] - frame["base"]
            if self.stack:
                self.stack[-1]["peak"] = max(self.stack[-1]["peak"], frame["peak"])
            if sink is not None:
                total = sink.setdefault(name, {"seconds": 0.0, "calls": 0, "peak": 0.0, "extra": 0.0})
                total["seconds"] += frame["seconds"]
                total["calls"] += 1
                total["peak"], total["extra"] = max(total["peak"], frame["peak"]), max(total["extra"], frame["extra"])

    def measure(self, fn, repeats: int = 1) -> tuple[object, dict]:
        """(result, measurements) of the fastest of `repeats` calls of `fn`"""
        best = None
        for _ in range(repeats):
            result = None  # (frees the last call's result first)
            with self.phase() as frame:
                result = fn()
            best = frame if best is None or frame["seconds"] < best["seconds"] else best
        return result, best


def random_vectors(params: dict[str, Tensor], n: int | None, seed: int = 0) -> dict[str, Tensor]:
    """Unit vectors {name: [n, *shape]} ([*shape] if n is None)"""
    g = torch.Generator().manual_seed(seed)
    V = {k: torch.randn(n or 1, *p.shape, generator=g).to(p) for k, p in params.items()}
    norms = torch.stack([v.flatten(1).pow(2).sum(1) for v in V.values()]).sum(0).sqrt()
    return {k: (v / norms.view(-1, *[1] * (v.ndim - 1)))[slice(None) if n else 0] for k, v in V.items()}


def make_attempt(meter: Meter, rows: list[dict]):
    """attempt(name, fn, repeats, **info): the result of `fn`, measured into a row of `rows` (None and an "error" row if out of memory)"""
    def attempt(name: str, fn, repeats: int = 1, **info):
        try:
            result, frame = meter.measure(fn, repeats)
        except torch.OutOfMemoryError:
            rows.append({"name": name, **info, "error": "out of memory"})
            gc.collect(), torch.cuda.empty_cache()
            return None
        rows.append({"name": name, **info, **{key: frame[key] for key in ("seconds", "peak", "base", "extra", "retained")}})
        return result
    attempt.rows = rows
    return attempt


def unit_costs(magic: Magic, X: Tensor, Y: Tensor, params: dict[str, Tensor], meter: Meter, max_k: int, compile: bool, rows: list[dict]):
    """Each operation of the step and its backward, alone, on the batch (X, Y), into `rows` (an operation out of memory: an "error" row)"""
    attempt = make_attempt(meter, rows)
    weights = torch.ones(len(X), device=X.device, requires_grad=True)
    forward_fn = partial(magic.objective.forward, X=X, Y=Y, weights=weights)
    params = {k: p.detach().requires_grad_() for k, p in params.items()}
    inputs = [*params.values(), weights]

    for _ in range(2):  # (warmup)
        torch.autograd.grad(torch.autograd.grad(forward_fn(params)[0], list(params.values()), create_graph=True)[0].sum(), inputs)
    with torch.no_grad():
        attempt("forward, no graph", lambda: forward_fn(params)[0], 3)
    loss = attempt("forward", lambda: forward_fn(params)[0], 3)
    attempt("backward", lambda: torch.autograd.grad(loss, params, retain_graph=True), 3)
    grads = attempt("backward, create_graph", lambda: torch.autograd.grad(loss, params, create_graph=True, retain_graph=True))
    del loss

    # LOBPCG's HVPs (torch.func, no graph), with the graph of the gradients alive as in the step
    for compiled in (False, True) if compile else (False,):
        func_hvp = hvp_from_forward(forward_fn, params, compiled)
        for n in (1, 2, 3, 4, 6):
            V = random_vectors(params, n)
            with torch.no_grad():
                if attempt("func HVP, first call", lambda: func_hvp(V), n=n, compiled=compiled) is not None:
                    attempt("func HVP", lambda: func_hvp(V), 3, n=n, compiled=compiled)
    hvp_fn = hvp_from_grads(grads, params)
    lobpcg_costs(forward_fn, params, hvp_fn, attempt, compile)

    # stateful HVPs through the graph of the gradients (MINRES: batched, no graph)
    for n in (1, 2, 4, 8, 16):
        V = random_vectors(params, n)
        attempt("graph HVP", lambda: hvp_fn(V), 3, n=n)

    # the forward's curvatures: H u with its graph, then a batched backward with n cotangents
    u = random_vectors(params, None)
    attempt("H u, no graph", lambda: hvp_fn(u), 3)
    H_u = attempt("H u, create_graph", lambda: hvp_fn(u, create_graph=True))
    for n in range(1, max_k + 1):
        others = list(random_vectors(params, n, seed=2).values())
        attempt("curvature backward", lambda: torch.autograd.grad(list(H_u.values()), params, others, retain_graph=True, is_grads_batched=True), 3, n=n)
    del H_u

    # a pair's curvature and its gradient by stateful autograd (STATEFUL), then by torch.func (the step, the eigenvector term) and its adjoint (the vjp)
    u, v, d_curvature_grad = (random_vectors(params, None, seed=seed) for seed in (3, 5, 4))
    attempt("pair, stateful", lambda: CentralFlowState._graph_curvature(LossGraph(params, hvp_fn), u, v), 3)
    loss = Loss(partial(magic.objective.forward, X=X, Y=Y), {k: p.detach() for k, p in params.items()}, {"weights": weights.detach()})
    pair_fn, vjp_fn = (COMPILED[fn] if compile else fn for fn in (curvature_grad, curvature_grad_vjp))
    with torch.no_grad():
        attempt("pair, func: first call", lambda: pair_fn(*loss, u, v))
        attempt("pair, func", lambda: pair_fn(*loss, u, v), 3)
        attempt("pair vjp, func: first call", lambda: vjp_fn(*loss, u, v, torch.ones((), device=X.device), d_curvature_grad))
        attempt("pair vjp, func", lambda: vjp_fn(*loss, u, v, torch.ones((), device=X.device), d_curvature_grad), 3)

    # the eigenvector term: H U with its graph (batched over k), then one backward per fn
    for n in (1, 2):
        U, Z = random_vectors(params, n, seed=5), random_vectors(params, n, seed=6)
        H_U = attempt("H U, create_graph", lambda: hvp_fn(U, create_graph=True), n=n)
        if H_U is not None:
            attempt("H U backward", lambda: torch.autograd.grad(list(H_U.values()), inputs, list(Z.values()), retain_graph=True, materialize_grads=True), 3, n=n)
        del H_U



def lobpcg_costs(forward_fn, params: dict[str, Tensor], graph_hvp, attempt, compile: bool):
    """LOBPCG from random vectors (4 and 9 tracked) with each HVP: torch.func eager / compiled in chunks of 3 vectors as the step,
    and the stateful `graph_hvp` on all the vectors at once; rows "LOBPCG, cold start" with `hvp` and `iters`"""
    device = next(iter(params.values())).device
    variants = [("func", partial(CentralFlowState._chunked_hvp, [(device, hvp_from_forward(forward_fn, params, False))]))]
    if compile:
        variants.append(("func, compiled", partial(CentralFlowState._chunked_hvp, [(device, hvp_from_forward(forward_fn, params, True))])))
    variants.append(("graph", graph_hvp))
    for tracked in (4, 9):
        for hvp, hvp_fn in variants:
            with torch.no_grad():
                result = attempt("LOBPCG, cold start", lambda: hessian_eigs(hvp_fn, random_vectors(params, tracked, seed=1), rtol=CentralFlowState.EIG_RTOL),
                                 n=tracked, hvp=hvp, compiled=hvp == "func, compiled")
            if result is not None:
                rows_last = attempt.rows[-1]
                rows_last["iters"] = result[2]
                rows_last["per_iter"] = rows_last["seconds"] / result[2]
            del result


def lobpcg_only(magic: Magic, X: Tensor, Y: Tensor, params: dict[str, Tensor], meter: Meter, compile: bool, rows: list[dict]):
    """The "lobpcg" section alone: the gradients' graph as in the step, then `lobpcg_costs`"""
    attempt = make_attempt(meter, rows)
    weights = torch.ones(len(X), device=X.device, requires_grad=True)
    forward_fn = partial(magic.objective.forward, X=X, Y=Y, weights=weights)
    params = {k: p.detach().requires_grad_() for k, p in params.items()}
    for _ in range(2):  # (warmup)
        torch.autograd.grad(torch.autograd.grad(forward_fn(params)[0], list(params.values()), create_graph=True)[0].sum(), [*params.values(), weights])
    grads = torch.autograd.grad(forward_fn(params)[0], params, create_graph=True)
    lobpcg_costs(forward_fn, params, hvp_from_grads(grads, params), attempt, compile)


class Phases:
    """Wall-clock marks at the end of the central flow phases, by monkeypatching CentralFlowState's methods: `times()` gives
    the duration of each phase since the previous mark (`start()`)"""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str):
        if self.cuda:
            torch.cuda.synchronize()
        self.marks.append((name, time.perf_counter()))

    def start(self):
        self.marks.clear()
        self.mark("start")

    def times(self, end: str) -> dict[str, float]:
        self.mark(end)
        return {name: t - prev for (_, prev), (name, t) in zip(self.marks, self.marks[1:])}

    def patch(self, monkeypatch: dict):
        """Mark the end of `_top_eigs` (lobpcg), `_pair_fns` (loss graphs), the curvatures of the step, `_flow_update`, the
        vjp's `_losses` (the backward to the curvatures before it) and `minres` (its start too: the pairs' adjoint before it)"""
        def after(name: str, fn, first_only: bool = False):
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                if not first_only or not any(n == name for n, _ in self.marks):
                    self.mark(name)
                return result
            return wrapper

        def before_and_after(name: str, fn):
            def wrapper(*args, **kwargs):
                self.mark(f"before {name}")
                result = fn(*args, **kwargs)
                self.mark(name)
                return result
            return wrapper

        originals = {name: getattr(CentralFlowState, name) for name in ("_top_eigs", "_pair_fns", "_pair_curvatures", "_flow_update", "_losses")}
        monkeypatch["CentralFlowState"] = originals
        monkeypatch["minres"] = second_order.minres
        CentralFlowState._top_eigs = after("lobpcg", originals["_top_eigs"])
        CentralFlowState._pair_fns = after("loss graphs", originals["_pair_fns"])
        CentralFlowState._pair_curvatures = after("curvatures", originals["_pair_curvatures"])
        CentralFlowState._flow_update = after("flow update", originals["_flow_update"])
        CentralFlowState._losses = after("to the curvatures", originals["_losses"], first_only=True)
        second_order.minres = before_and_after("minres", second_order.minres)

    @staticmethod
    def unpatch(monkeypatch: dict):
        for name, fn in monkeypatch["CentralFlowState"].items():
            setattr(CentralFlowState, name, fn)
        second_order.minres = monkeypatch["minres"]



def step_costs(magic: Magic, X: Tensor, Y: Tensor, params: dict[str, Tensor], meter: Meter, k: int, n_steps: int,
               helpers: tuple[str, ...], compile: bool, rows: list[dict]):
    """`n_steps` steps from `params` with the lr of exactly `k` critical eigenvalues and the vjp of the last one, by phase, eager
    and (`compile`) with the compiled HVPs, into `rows`"""
    device = X.device
    forward_fn = partial(magic.objective.forward, X=X, Y=Y)
    phases = Phases(meter.cuda)
    with torch.no_grad():
        eigvals, eigvecs, _ = hessian_eigs(partial(CentralFlowState._chunked_hvp, [(device, hvp_from_forward(forward_fn, params))]),
                                           random_vectors(params, max(k + 1, 4), seed=1), rtol=CentralFlowState.EIG_RTOL)
    lr = CentralFlowState.THRESHOLD / (eigvals[k - 1] * eigvals[k]).sqrt().item()  # exactly k eigenvalues above THRESHOLD / lr
    click.echo(f"  top eigenvalues {[f'{v:.3g}' for v in eigvals.tolist()]}: lr {lr:.3g}")

    def measured(name: str, fn, end: str, **info) -> bool:
        """Run and record `fn` with its phases (False if it ran out of memory)"""
        phases.start()
        try:
            with meter.phase() as m:
                result = fn()
        except torch.OutOfMemoryError as error:
            click.echo(f"  {name}: out of memory ({str(error)[:120]})")
            rows.append({"call": name, **info, "error": "out of memory"})
            return False
        by_phase = phases.times(end)
        rows.append({"call": name, **info, "seconds": m["seconds"], "peak": m["peak"], "phases": by_phase, "result": result})
        click.echo(f"  {name}: {m['seconds']:.2f} s, peak {m['peak']:.2f} GB" + ("" if result is None else f"; {result}")
                   + "\n    phases: " + ", ".join(f"{p} {s:.2f}" for p, s in by_phase.items()))
        return True

    monkeypatch: dict = {}
    phases.patch(monkeypatch)
    try:
        for compiled in (False, True) if compile else (False,):
            click.echo(f"  --- compile {compiled}")
            state = CentralFlowState(params={name: p.clone() for name, p in params.items()}, lr=LRSchedule(lr), eigvecs=eigvecs,
                                     compile=compiled, devices=helpers)
            previous = state
            for t in range(n_steps):
                def step():
                    nonlocal state, previous
                    previous = state.apply(lambda p: p.detach().requires_grad_())
                    state = previous.step(t, forward_fn)[0].apply(lambda p: p.detach())
                    return f"{int(state.stats['n_critical'])} critical, {int(state.stats['eig_iters'])} LOBPCG iterations"
                if not measured(f"step {t}", step, end="rest", compiled=compiled):
                    break
            weights = torch.ones(len(X), device=device, requires_grad=True)  # the vjp of the last step: from its state, with the next state's aux
            d_state = state.apply(lambda p: torch.randn(1, *p.shape, generator=torch.Generator().manual_seed(0)).to(p))
            measured(f"vjp of step {n_steps - 1}", lambda: previous.vjp(n_steps - 1, partial(forward_fn, weights=weights), [*previous, weights], d_state, state) and None,
                     end="eigenvector term", compiled=compiled)
            del previous, state
            if meter.cuda:
                torch.cuda.empty_cache()
    finally:
        Phases.unpatch(monkeypatch)


# fixed colors by what a line or a bar segment stands for (the unit operations, then the step phases)
COLORS = {"func HVP": "#2a78d6", "func HVP, compiled": "#eb6834", "graph HVP": "#1baf7a", "curvature backward": "#eda100", "H U backward": "#e87ba4",
          "lobpcg": "#2a78d6", "to the curvatures": "#eda100", "curvatures": "#eb6834", "before minres": "#1baf7a", "minres": "#6250d6",
          "eigenvector term": "#e87ba4", "loss graphs": "#008300", "flow update": "#e34948", "rest": "#9a9a9a"}
STEP_PHASES = ["lobpcg", "loss graphs", "to the curvatures", "curvatures", "flow update", "rest", "before minres", "minres", "eigenvector term"]


def markdown_table(header: list[str], rows: list[list]) -> str:
    """Numbers to 3 significant digits, None to blank"""
    def cell(value) -> str:
        if value is None:
            return ""
        return f"{value:.3g}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]) + "\n"


def unit_rows(rows: list[dict], name: str, compiled: bool | None = None) -> dict[int | None, dict]:
    """The unit rows of `name` by their n (None: no n), the compiled ones if asked"""
    return {row.get("n"): row for row in rows
            if row["section"] == "unit" and row["name"] == name and (compiled is None or row.get("compiled", False) == compiled)}


def unit_tables(rows: list[dict]) -> str:
    def seconds(selected: dict, n, scale: float = 1000.0):
        row = selected.get(n)
        return None if row is None or "error" in row else row["seconds"] * scale

    def peak(selected: dict, n):
        row = selected.get(n)
        return None if row is None or "error" in row else row["peak"]

    text = "## Single operations\n\n"
    basics = [row for row in rows if row["section"] == "unit" and "n" not in row and "compiled" not in row]
    text += markdown_table(["operation", "ms", "peak GB", "extra GB", "retained GB"],
                           [[r["name"], *([None] * 4 if "error" in r else [r["seconds"] * 1000, r["peak"], r["extra"], r["retained"]])] for r in basics])
    by_n = {"func HVP": unit_rows(rows, "func HVP", False), "func HVP, compiled": unit_rows(rows, "func HVP", True),
            "compile (first call)": unit_rows(rows, "func HVP, first call", True), "graph HVP": unit_rows(rows, "graph HVP"),
            "curvature backward": unit_rows(rows, "curvature backward"), "H U, create_graph": unit_rows(rows, "H U, create_graph"),
            "H U backward": unit_rows(rows, "H U backward")}
    ns = sorted({n for selected in by_n.values() for n in selected})
    text += "\n## Batched operations by number of vectors n\n\nMilliseconds (the compile column: seconds of the first call, which compiles):\n\n"
    text += markdown_table(["n", *by_n], [[n, *(seconds(selected, n, 1.0 if name.startswith("compile") else 1000.0) for name, selected in by_n.items())] for n in ns])
    text += "\nPeak memory, GB:\n\n"
    text += markdown_table(["n", *by_n], [[n, *(peak(selected, n) for selected in by_n.values())] for n in ns])
    lobpcg = [row for row in rows if row["section"] == "unit" and row["name"] == "LOBPCG, cold start"]
    text += "\n## LOBPCG from random vectors\n\nHVP: func = torch.func forward-over-backward in chunks of 3 vectors (as the step), graph = stateful autograd through the gradients' graph, all the vectors at once\n\n"
    text += markdown_table(["vectors", "HVP", "iterations", "s", "s / iteration", "peak GB"],
                           [[r["n"], r.get("hvp", "func, compiled" if r.get("compiled") else "func"), r.get("iters"),
                             *([None] * 3 if "error" in r else [r["seconds"], r.get("per_iter"), r["peak"]])] for r in lobpcg])
    return text


def steps_table(rows: list[dict]) -> str:
    rows = [row for row in rows if row["section"] == "steps"]
    phases = [p for p in STEP_PHASES if any(p in row.get("phases", {}) for row in rows)]
    text = "## Steps and vjp by phase\n\nSeconds; the first compiled step includes the compilation of the HVPs.\n\n"
    return text + markdown_table(["k", "compiled", "call", "s", "peak GB", *phases, "result"],
                                 [[r["k"], r.get("compiled", False), r.get("call", ""), *([None] * (2 + len(phases)) + [r["error"]] if "error" in r else
                                   [r["seconds"], r["peak"], *(r["phases"].get(p) for p in phases), r.get("result") or ""])] for r in rows])


def plot_unit(rows: list[dict], title: str, path: Path) -> bool:
    """Time and peak memory of the batched operations against the number of vectors (False: none measured)"""
    series = {"func HVP": unit_rows(rows, "func HVP", False), "func HVP, compiled": unit_rows(rows, "func HVP", True), "graph HVP": unit_rows(rows, "graph HVP"),
              "curvature backward": unit_rows(rows, "curvature backward"), "H U backward": unit_rows(rows, "H U backward")}
    series = {name: {n: r for n, r in selected.items() if "error" not in r} for name, selected in series.items() if selected}
    if not series:
        return False
    fig, (ax_time, ax_memory) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, selected in series.items():
        ns = sorted(selected)
        ax_time.plot(ns, [selected[n]["seconds"] for n in ns], "o-", ms=5, lw=1.5, color=COLORS[name], label=name)
        ax_memory.plot(ns, [selected[n]["peak"] for n in ns], "o-", ms=5, lw=1.5, color=COLORS[name], label=name)
    for ax, ylabel in ((ax_time, "seconds"), (ax_memory, "peak GB")):
        ax.set_xlabel("vectors n")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    ax_time.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_steps(rows: list[dict], title: str, path: Path):
    """Stacked phases of each step and vjp, grouped by k (eager and compiled), and their peak memory"""
    rows = [row for row in rows if row["section"] == "steps" and "error" not in row]
    groups = sorted({(row["k"], row.get("compiled", False)) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, is_vjp in zip(axes[:2], (False, True)):
        shown_phases, position, ticks = set(), 0.0, []
        for k, compiled in groups:
            calls = [row for row in rows if (row["k"], row.get("compiled", False)) == (k, compiled) and row["call"].startswith("vjp") == is_vjp]
            if compiled and not is_vjp:
                calls = calls[1:] or calls  # (the first compiled step is dominated by the compilation)
            start = position
            for row in calls:
                bottom = 0.0
                for phase in STEP_PHASES:
                    if phase in row["phases"]:
                        ax.bar(position, row["phases"][phase], 0.8, bottom=bottom, color=COLORS[phase], edgecolor="white", lw=0.5,
                               label=phase if phase not in shown_phases else None)
                        shown_phases.add(phase)
                        bottom += row["phases"][phase]
                position += 1
            ticks.append(((start + position - 1) / 2, f"k={k}\n{'compiled' if compiled else 'eager'}"))
            position += 0.6
        ax.set_xticks([t for t, _ in ticks], [label for _, label in ticks], fontsize=8)
        ax.set_ylabel("seconds")
        ax.set_title("vjp of the last step" if is_vjp else "steps (compiled: after the first)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes[2]
    for is_vjp, marker, label in ((False, "o", "step"), (True, "s", "vjp")):
        xs = [i for i, g in enumerate(groups) for row in rows if (row["k"], row.get("compiled", False)) == g and row["call"].startswith("vjp") == is_vjp]
        ys = [row["peak"] for g in groups for row in rows if (row["k"], row.get("compiled", False)) == g and row["call"].startswith("vjp") == is_vjp]
        ax.plot(xs, ys, marker, ms=6, color=COLORS["minres" if is_vjp else "lobpcg"], label=label, alpha=0.8)
    ax.set_xticks(range(len(groups)), [f"k={k}\n{'compiled' if c else 'eager'}" for k, c in groups], fontsize=8)
    ax.set_ylabel("peak GB")
    ax.set_title("peak memory")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(results: dict, out_dir: Path, tag: str):
    """TAG.md, TAG_unit.png and TAG_steps.png of `results` in `out_dir`"""
    rows = results["rows"]
    title = (f"{results['gpu']}: {results['model']} width {results['width']} ({results['n_params'] / 1000:.1f}k params), batch {results['B']}"
             + (f", helpers {list(results['devices'])}" if results["devices"] else ""))
    text = f"# Central flow profile `{tag}`\n\n{title}\n\n"
    errors = [row for row in rows if "error" in row]
    if errors:
        text += "Errors: " + "; ".join(f"{row.get('section')} {row.get('name', row.get('call', ''))} k={row.get('k', '')}: {row['error']}" for row in errors) + "\n\n"
    if any(row["section"] == "unit" for row in rows):
        if plot_unit(rows, title, out_dir / f"{tag}_unit.png"):
            text += f"![]({tag}_unit.png)\n\n"
        text += unit_tables(rows)
    if any(row["section"] == "steps" for row in rows):
        plot_steps(rows, title, out_dir / f"{tag}_steps.png")
        text += f"\n![]({tag}_steps.png)\n\n" + steps_table(rows)
    (out_dir / f"{tag}.md").write_text(text)


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--tag", default=None, help="name of the output files in DIR/profile (default: the GPU)")
@click.option("--report", type=click.Path(path_type=Path, dir_okay=False, exists=True), default=None, help="only report an existing results JSON")
@click.option("--section", type=click.Choice(["unit", "lobpcg", "steps", "all"]), default="all", show_default=True)
@click.option("--n-train", type=int, default=None, help="examples in the batch (default: the config's n_train)")
@click.option("--ks", default="1,2,4,8", show_default=True, help="critical eigenvalues of the steps section (comma-separated)")
@click.option("--steps", type=int, default=1, show_default=True, help="consecutive steps per k")
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--devices", type=str, default="", help="comma-separated helper devices (default: none)")
@click.option("--compile/--no-compile", default=True, help="also the compiled HVPs of LOBPCG (needs a C++ compiler: not on the login node)")
def profile(exp_dir: Path, tag: str | None, report: Path | None, section: str, n_train: int | None, ks: str, steps: int, device: str, devices: str, compile: bool):
    out_dir = exp_dir / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    if report is not None:
        results = json.loads(report.read_text())
        tag = tag or re.sub(r"[^a-z0-9]+", "_", results["gpu"].lower()).strip("_")
        (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=1))
        write_report(results, out_dir, tag)
        click.echo(f"report of {report}: {out_dir / tag}.{{json,md}}")
        return
    torch._dynamo.config.recompile_limit = 64  # (the compiled HVPs of several sizes)
    exp = Experiment(exp_dir, device)
    n = len(exp.train_ds) if n_train is None else n_train
    X, Y = (t.to(device) for t in exp.train_ds.examples(torch.arange(n)))
    params = {name: p.detach().clone() for name, p in exp.init_state.params.items()}
    helpers = tuple(d for d in devices.split(",") if d)
    meter = Meter(device)
    results = {"gpu": torch.cuda.get_device_name() if meter.cuda else "cpu", "model": exp.cfg.model, "width": exp.cfg.width,
               "n_params": sum(p.numel() for p in params.values()), "B": n, "devices": helpers, "rows": []}
    if tag is None:
        gpu = re.search(r"(a100|h100|l4|t4|v100)", results["gpu"].lower())
        tag = (gpu.group(1) if gpu else re.sub(r"[^a-z0-9]+", "_", results["gpu"].lower()).strip("_")) + ("" if compile else "_eager")
    out = out_dir / f"{tag}.json"
    click.echo(f"{exp.cfg.model} width {exp.cfg.width}: {results['n_params'] / 1000:.1f}k params, batch of {n} on {device}"
               + (f", helper devices {helpers}" if helpers else ""))
    k_list = [int(k) for k in ks.split(",")]

    def run(name: str, fn, **key):
        """Append the rows of `fn()` (or its error) and save"""
        rows: list[dict] = []
        try:
            fn(rows)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            rows.append({"error": str(error)[:300]})
        gc.collect()
        if meter.cuda:
            torch.cuda.empty_cache()
        results["rows"] += [{"section": name, **key, **row} for row in rows]
        out.write_text(json.dumps(results, indent=1, default=str))
        write_report(results, out_dir, tag)
        click.echo(f"{name} {key}: {len(rows)} rows" + "".join(f" ERROR {row['error']}" for row in rows if "error" in row))

    if section in ("unit", "all"):
        run("unit", lambda rows: unit_costs(exp.magic, X, Y, params, meter, max(k_list), compile, rows))
    if section == "lobpcg":
        run("unit", lambda rows: lobpcg_only(exp.magic, X, Y, params, meter, compile, rows))
    if section in ("steps", "all"):
        for k in k_list:
            run("steps", lambda rows: step_costs(exp.magic, X, Y, params, meter, k, steps, helpers, compile, rows), k=k)


if __name__ == "__main__":
    profile()

"""The optimizer states whose step uses Hessian-vector products (`SecondOrderState`): sharpness-aware, Edge Flow, central flow and Newton steps."""

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, ClassVar, Mapping, NamedTuple, Self, Sequence

import torch
from torch import Tensor
from torch.func import grad, grad_and_value, jvp

from src.minres import minres
from src.lobpcg import flatten, hessian_eigs, unflatten
from src.objective import HVP, hvp_from_forward, hvp_from_grads
from src.optimizers import Forward, OptimState, StepOutput, dict_zip
from src.sdcp import sdcp


class LossGraph(NamedTuple):
    """The gradients' graph of a step's loss on one device"""

    params: dict[str, Tensor]
    hvp_fn: HVP

    @property
    def device(self) -> torch.device:
        return next(iter(self.params.values())).device


class Loss(NamedTuple):
    """A step's loss `forward_fn(params, **inputs)[0]` on one device (detached tensors)"""

    forward_fn: Callable
    params: dict[str, Tensor]
    inputs: dict[str, Tensor]

    @property
    def device(self) -> torch.device:
        return next(iter(self.params.values())).device


def curvature(forward_fn: Callable, params: dict[str, Tensor], inputs: dict[str, Tensor], u: dict[str, Tensor], v: dict[str, Tensor]) -> Tensor:
    """u^T H v of the loss `forward_fn(params, **inputs)[0]`, in forward mode"""
    loss_fn = lambda p: forward_fn(p, **inputs)[0]
    return jvp(lambda p: jvp(loss_fn, (p,), (v,))[1], (params,), (u,))[1]


def curvature_grad(forward_fn: Callable, params: dict[str, Tensor], inputs: dict[str, Tensor], u: dict[str, Tensor], v: dict[str, Tensor]):
    """(d(u^T H v)/d(params, inputs), u^T H v)"""
    return grad_and_value(lambda p, x: curvature(forward_fn, p, x, u, v), argnums=(0, 1))(params, inputs)


def curvature_grad_vjp(
    forward_fn: Callable, params: dict[str, Tensor], inputs: dict[str, Tensor], u: dict[str, Tensor], v: dict[str, Tensor],
    d_curvature: Tensor, d_curvature_grad: dict[str, Tensor],
):
    """Adjoint of u^T H v and its gradient to the params at the cotangents `d_curvature`, `d_curvature_grad`: d/d(params, inputs, u, v)"""
    def cotangent_product(p, x, u, v):
        value, directional = jvp(lambda q: curvature(forward_fn, q, x, u, v), (p,), (d_curvature_grad,))
        return d_curvature * value + directional

    return grad(cotangent_product, argnums=(0, 1, 2, 3))(params, inputs, u, v)


COMPILED = {fn: torch.compile(fn) for fn in (curvature_grad, curvature_grad_vjp)}  # (module-level: one cache for every step)
PairFn = Callable[[dict[str, Tensor], dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]  # (u, v) -> (u^T H v, its gradient to the params)


@dataclass(kw_only=True)
class SecondOrderState(OptimState):
    aux = ("stats",)
    stats: dict[str, Tensor] = field(default_factory=dict)

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """The grads always have their graph (for the HVPs); `create_graph` is of the HVPs"""
        loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        return self._update(t, grads, create_graph), grads, loss, out

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        raise NotImplementedError

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SharpnessState(SecondOrderState):
    """SGD with step size `lr * 2 / sharpness`.
    Sharpness is computed with its graph, allowing backprop through a step"""

    aux = ("eigvec", "stats")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # top eigenvector (warm start)

    def __post_init__(self):
        super().__post_init__()
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            self.eigvec = {
                k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if not self.stats:
            p = next(iter(self.params.values()))
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "eig_iters": torch.zeros_like(nan)}

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        hvp_fn = hvp_from_grads(grads, self.params)
        sharpness, eigvec, eig_iters = hessian_eigs(hvp_fn, self.eigvec, batched=False)
        if create_graph:
            H_eigvec = hvp_fn(eigvec, create_graph=True)
            sharpness = torch.stack(  # S = v^T H v
                [(v * Hv).sum() for v, Hv in dict_zip(eigvec, H_eigvec).values()]
            ).sum()
        step_size = self.lr(t) * 2 / sharpness
        params = {
            k: p - step_size * (g + self.wd * p)
            for k, (p, g) in dict_zip(self.params, grads).items()
        }
        stats = {"sharpness": sharpness.detach(), "eig_iters": torch.full_like(sharpness.detach(), eig_iters)}
        return replace(self, params=params, eigvec=eigvec, stats=stats)

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """The step size with the last step's sharpness (0 before the first step)"""
        return {k: self.lr(t) * 2 / self.stats["sharpness"] for k in self.params}


@dataclass(kw_only=True)
class EdgeFlowState(SecondOrderState):
    """Edge Flow (Marion 2026, Algorithm 1): `lr` is the modeled GD learning rate eta
    update is 1/2 * (grad(params + eps * eigvec) + grad(params - eps * eigvec))
    `eigvec` (a power iteration) and `eps` (grows iff sharpness > 2 / eta)"""

    SUBSTEPS: ClassVar[int] = 4
    MIN_EPS: ClassVar[float] = 1e-4

    groups = ("params", "eigvec", "eps")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # None -> random unit
    eps: dict[str, Tensor] = field(default_factory=dict)  # {"eps": []}; None -> MIN_EPS

    def __post_init__(self):
        super().__post_init__()
        p = next(iter(self.params.values()))
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            noise = {k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for k, p in self.params.items()}
            norm = torch.sqrt(sum((v * v).sum() for v in noise.values()))
            self.eigvec = {k: v / norm for k, v in noise.items()}
        if not self.eps:
            self.eps = {"eps": torch.full((), self.MIN_EPS, dtype=p.dtype, device=p.device)}
        if not self.stats:
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "eps": self.eps["eps"].detach(), "center_loss": nan}

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """The loss and grads are averaged over the endpoints `params` +- eps * eigvec; the outputs are at `params`"""
        eps = self.eps["eps"]
        left, right = [
            {k: p + sign * eps * v for k, (p, v) in dict_zip(self.params, self.eigvec).items()} for sign in (1, -1)
        ]
        loss = (forward_fn(left)[0] + forward_fn(right)[0]) / 2
        center_loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=create_graph)
        center_grads = torch.autograd.grad(center_loss, self.params, create_graph=True)

        H_eigvec = hvp_from_grads(center_grads, self.params)(self.eigvec, create_graph=create_graph)
        sharpness = torch.stack([hv.pow(2).sum() for hv in H_eigvec.values()]).sum().sqrt()  # |H eigvec|: keeps eigvec unit
        rho = self.lr(t) / self.SUBSTEPS  # Edge-Flow step size
        params = {k: p - rho * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}
        eigvec = {k: hv / sharpness for k, hv in H_eigvec.items()}
        eps = (eps * (1 + rho * (sharpness - 2 / self.lr(t)))).clamp_min(self.MIN_EPS)

        stats = {"sharpness": sharpness.detach(), "eps": eps.detach(), "center_loss": center_loss.detach()}
        new_state = replace(self, params=params, eigvec=eigvec, eps={"eps": eps}, stats=stats)
        return new_state, center_grads, loss, out

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        return {k: self.lr(t) / self.SUBSTEPS for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """2 / eta on the scale of the step rho"""
        return 2 / self.SUBSTEPS


@dataclass(kw_only=True)
class CentralFlowState(SecondOrderState):
    """Central flow of GD with learning rate `lr` (Cohen et al. 2024, arXiv:2410.24206)
        params <- params - lr / SUBSTEPS * (grads + 1/2 * d<Sigma, H>/d(params))
    Sigma = U X U^T, the covariance of GD's oscillations: U the critical eigenvectors,
    X the `sdcp` solution that keeps U^T H U <= 2 / lr after the step"""

    SUBSTEPS: ClassVar[int] = 4
    N_EIGS: ClassVar[int] = 8  # at most this many critical eigenvalues
    THRESHOLD: ClassVar[float] = 1.95  # an eigenvalue is critical above THRESHOLD / lr
    EIG_RTOL: ClassVar[float] = 1e-3  # of LOBPCG
    VJP_RTOL: ClassVar[float] = 1e-3  # of MINRES
    STATEFUL: ClassVar[bool] = False  # the step's third derivatives by stateful autograd through the gradients' graph (default: torch.func)

    aux = ("eigvecs", "curvatures", "curvature_grads", "stats")
    eigvecs: dict[str, Tensor] = field(default_factory=dict)  # {name: [n, *shape]}, critical ones first
    curvatures: dict[str, Tensor] = field(default_factory=dict)  # {"curvatures": [k, k]} u_i^T H u_j
    curvature_grads: dict[str, Tensor] = field(default_factory=dict)  # {name: [k, k, *shape]}
    compile: bool = False  # torch.compile the torch.func derivatives (HVPs, third and fourth)
    devices: tuple[str, ...] = ()  # helper devices for LOBPCG's HVPs and pairs of critical eigenvectors

    def __post_init__(self):
        super().__post_init__()
        if not self.eigvecs:
            g = torch.Generator().manual_seed(0)
            self.eigvecs = {
                k: torch.randn(2, *p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if self.compile:  # the HVPs are compiled once per device and number of vectors (1, 2, 3)
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 3 * (1 + len(self.devices))
            )

    @property
    def n_critical(self) -> int:
        """The number of critical eigenvalues of the last step"""
        return len(self.curvatures["curvatures"]) if self.curvatures else 0

    @property
    def critical(self) -> dict[str, Tensor]:
        """The critical eigenvectors of the last step {name: [k, *shape]}"""
        return {k: v[: self.n_critical] for k, v in self.eigvecs.items()}

    def _warm_start(self, eigvecs: dict[str, Tensor], n_critical: int) -> dict[str, Tensor]:
        """Add an eigenvector (if needed) for the next step"""
        n_old = len(next(iter(eigvecs.values())))
        n_new = min(max(n_critical + 1, 2), self.N_EIGS + 1) - n_old
        if n_new <= 0:
            return {k: v[: n_old + n_new] for k, v in eigvecs.items()}
        g = torch.Generator().manual_seed(0)
        return {
            k: torch.cat([v, torch.randn(n_new, *v.shape[1:], generator=g).to(v.device, v.dtype)])
            for k, v in eigvecs.items()
        }

    # Helper devices: the HVPs (`_chunked_hvp`) and the pairs of critical eigenvectors are dealt among the loss's copies on them.

    @staticmethod
    def _dealt(n_items: int, n_devices: int) -> list[tuple[int, list[int]]]:
        """(device, its items) round-robin, this device (0) last
        so that nothing is moved to a helper once it is busy"""
        return [(d, list(range(d, n_items, n_devices))) for d in reversed(range(n_devices))]

    def _on_device(
        self, forward_fn: Forward, device: str, inputs: Sequence[Tensor] = ()
    ) -> tuple[Forward, dict[str, Tensor], list[Tensor]]:
        """`forward_fn`, the params and `inputs` copied to `device`"""
        assert isinstance(forward_fn, partial), "helper devices need to move the batch"
        copies = {id(x): x.detach().to(device).requires_grad_() for x in (*self.params.values(), *inputs)}
        move = lambda x: copies.get(id(x), x.detach().to(device)) if isinstance(x, Tensor) else x
        params = {k: move(p) for k, p in self.params.items()}
        device_forward = partial(
            forward_fn.func,
            *map(move, forward_fn.args),
            **{k: move(x) for k, x in forward_fn.keywords.items()},
        )
        return device_forward, params, [copies[id(x)] for x in inputs]

    def _hvp_fns(self, forward_fn: Forward) -> list[tuple[torch.device | str, HVP]]:
        """H V by torch.func (no graph) on this device, then on each helper device"""
        hvp_fns = [
            (next(iter(self.params.values())).device,
            hvp_from_forward(forward_fn, self.params, self.compile)),
        ]
        for device in self.devices:
            device_forward, params, _ = self._on_device(forward_fn, device)
            hvp_fns.append((device, hvp_from_forward(device_forward, params, self.compile)))
        return hvp_fns

    def _loss_graphs(self, forward_fn: Forward, grads: dict[str, Tensor]) -> list[LossGraph]:
        """The graph of `grads` (for HVPs with a graph), then its copy on each helper device"""
        graphs = [LossGraph(self.params, hvp_from_grads(grads, self.params))]
        for device in self.devices:
            device_forward, params, _ = self._on_device(forward_fn, device)
            device_grads = torch.autograd.grad(device_forward(params)[0], params, create_graph=True)
            graphs.append(LossGraph(params, hvp_from_grads(device_grads, params)))
        return graphs

    def _losses(self, forward_fn: Forward, inputs: Sequence[Tensor] = ()) -> list[Loss]:
        """The loss as a function of the params and `inputs` (keyword tensors of `forward_fn`), then its copy on each helper device"""
        copies = [(forward_fn, self.params, inputs), *(self._on_device(forward_fn, device, inputs) for device in self.devices)]
        losses = []
        for device_forward, params, device_inputs in copies:
            if inputs:  # the keywords of the inputs become arguments
                keywords = {name: x for x in device_inputs for name, y in device_forward.keywords.items() if y is x}
                bound = {name: y for name, y in device_forward.keywords.items() if name not in keywords}
                device_forward = partial(device_forward.func, *device_forward.args, **bound)
            else:
                keywords = {}
            losses.append(Loss(
                device_forward, {k: p.detach() for k, p in params.items()}, {k: x.detach() for k, x in keywords.items()}
            ))
        return losses

    @classmethod
    def _chunked_hvp(
        cls, hvp_fns: Sequence[tuple[torch.device | str, HVP]], V: dict[str, Tensor], create_graph: bool = False
    ) -> dict[str, Tensor]:
        """H V in calls of 3 vectors (the last one of the rest: 1 or 2) dealt among `hvp_fns`"""
        chunks = [dict(zip(V, chunk)) for chunk in zip(*(v.split(3) for v in V.values()))]
        results: dict[int, dict[str, Tensor]] = {}
        for d, items in cls._dealt(len(chunks), len(hvp_fns)):
            device, hvp_fn = hvp_fns[d]
            for c in items:
                results[c] = hvp_fn({k: v.to(device) for k, v in chunks[c].items()})
        return {k: torch.cat([results[c][k].to(v.device) for c in range(len(chunks))]) for k, v in V.items()}

    # The forward step

    def _top_eigs(self, t: int, forward_fn: Forward) -> tuple[Tensor, dict[str, Tensor], int, int]:
        """(tracked eigenvalues, their eigenvectors, LOBPCG iterations, number of critical eigenvalues)"""
        with torch.no_grad():  # (`forward_fn` may hold tensors that require grad)
            hvp_fn = partial(self._chunked_hvp, self._hvp_fns(forward_fn))
            eigvals, eigvecs, eig_iters = hessian_eigs(hvp_fn, self.eigvecs, rtol=self.EIG_RTOL)
        n_critical = int((eigvals + self.wd > self.THRESHOLD / self.lr(t)).sum())
        assert n_critical <= self.N_EIGS, f"step {t}: more than N_EIGS = {self.N_EIGS} critical eigenvalues"
        return eigvals, eigvecs, eig_iters, n_critical

    @staticmethod
    def _graph_curvature(graph: LossGraph, u: dict[str, Tensor], v: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """u^T H v [] and its gradient {name: [*shape]} by stateful autograd, on the graph's device"""
        H_v = graph.hvp_fn(v, create_graph=True)
        curvature = torch.stack([(u[k] * H_v[k]).sum() for k in H_v]).sum()
        curvature_grad = torch.autograd.grad(
            list(H_v.values()), list(graph.params.values()), [u[k] for k in H_v], retain_graph=True
        )
        return curvature.detach(), dict(zip(graph.params, curvature_grad))

    def _func_curvature(self, loss: Loss, u: dict[str, Tensor], v: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """u^T H v [] and its gradient {name: [*shape]} by torch.func, on the loss's device"""
        fn = COMPILED[curvature_grad] if self.compile else curvature_grad
        with torch.no_grad():  # (`forward_fn` may hold tensors that require grad)
            (curvature_grads, _), curvature_value = fn(*loss, u, v)
        return curvature_value, curvature_grads

    def _pair_fns(self, forward_fn: Forward, grads: dict[str, Tensor]) -> list[tuple[torch.device, PairFn]]:
        """(u, v) -> (u^T H v, its gradient) on this device, then on each helper device"""
        if self.STATEFUL:
            return [(graph.device, partial(self._graph_curvature, graph)) for graph in self._loss_graphs(forward_fn, grads)]
        return [(loss.device, partial(self._func_curvature, loss)) for loss in self._losses(forward_fn)]

    @staticmethod
    def _symmetric(pairs: Tensor, k: int) -> Tensor:
        """[#pairs, ...] over the pairs i <= j in row order -> [k, k, ...] with [i, j] = [j, i]"""
        rows, cols = torch.triu_indices(k, k, device=pairs.device)
        symmetric = pairs.new_zeros(k, k, *pairs.shape[1:])
        symmetric[rows, cols] = symmetric[cols, rows] = pairs
        return symmetric

    def _pair_curvatures(
        self, pair_fns: Sequence[tuple[torch.device, PairFn]], critical: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """u_i^T H u_j [k, k] and their gradients {name: [k, k, *shape]}, one pair at a time among `pair_fns`"""
        k = len(next(iter(critical.values())))
        rows, cols = torch.triu_indices(k, k)
        results: dict[int, tuple[Tensor, dict[str, Tensor]]] = {}
        for d, items in self._dealt(len(rows), len(pair_fns)):
            device, pair_fn = pair_fns[d]
            U = {name: u.to(device) for name, u in critical.items()}
            for n in items:
                i, j = int(rows[n]), int(cols[n])
                results[n] = pair_fn({name: u[i] for name, u in U.items()}, {name: u[j] for name, u in U.items()})
        device = pair_fns[0][0]
        curvatures = [results[n][0].to(device) for n in range(len(rows))]
        curvature_grads = {name: [results[n][1][name].to(device) for n in range(len(rows))] for name in critical}
        return (
            self._symmetric(torch.stack(curvatures), k),
            {name: self._symmetric(torch.stack(curvature_grads[name]), k) for name in critical},
        )

    def _flow_update(
        self, t: int, grads: dict[str, Tensor], curvatures: Tensor, curvature_grads: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """A central flow step `loss_grads + <curvature_grads, X> / 2`, where X solves the SDCP"""
        lr, step_size = self.lr(t), self.lr(t) / self.SUBSTEPS
        k = len(curvatures)
        reg_grads = {name: g + self.wd * p for name, (p, g) in dict_zip(self.params, grads).items()}
        flat_grads = {name: C_grad.flatten(2) for name, C_grad in curvature_grads.items()}  # [k, k, #params]
        drift = torch.stack([  # [k, k]
            C_grad @ g.flatten() for C_grad, g in dict_zip(flat_grads, reg_grads).values()
        ]).sum(0)
        gram = torch.stack([  # [k, k, k, k]
            torch.einsum("ijp,klp->ijkl", C_grad, C_grad) for C_grad in flat_grads.values()
        ]).sum(0)
        edge = torch.eye(k).to(curvatures) * (2 / lr - self.wd)
        X = sdcp(edge - (curvatures - step_size * drift), step_size / 2 * gram)
        params = {
            name: p - step_size * (g + torch.tensordot(X, C_grad, 2) / 2)
            for name, (p, g, C_grad) in dict_zip(self.params, reg_grads, curvature_grads).items()
        }
        return params, X

    def _gradient_flow_step(self, t: int, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """A gradient flow step without critical eigenvectors"""
        step_size = self.lr(t) / self.SUBSTEPS
        return {k: p - step_size * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        assert not create_graph
        loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        eigvals, eigvecs, eig_iters, n_critical = self._top_eigs(t, forward_fn)
        if n_critical == 0:
            curvatures, curvature_grads, X = eigvals.new_zeros(0, 0), {}, torch.zeros(0, 0)
            params = self._gradient_flow_step(t, grads)
        else:
            critical = {k: v[:n_critical] for k, v in eigvecs.items()}
            curvatures, curvature_grads = self._pair_curvatures(self._pair_fns(forward_fn, grads), critical)
            params, X = self._flow_update(t, grads, curvatures, curvature_grads)
        stats = {f"eig/{i + 1}": eig for i, eig in enumerate(eigvals)} | {
            "eig_iters": torch.full_like(eigvals[0], eig_iters),
            "n_critical": torch.full_like(eigvals[0], n_critical),
            "oscillation": X.detach().trace().to(eigvals),  # GD's mean loss is ~ loss + oscillation / lr
        }
        return replace(
            self, params=params, eigvecs=self._warm_start(eigvecs, n_critical),
            curvatures={"curvatures": curvatures},
            curvature_grads=curvature_grads, stats=stats,
        ), grads, loss, out

    # The backward step. The cotangents are batched over the fns (the eval sets): [#fns, ...]

    def _pair_curvatures_vjp(
        self, losses: Sequence[Loss], critical: dict[str, Tensor],
        d_curvatures: Tensor, d_curvature_grads: dict[str, Tensor],
    ) -> tuple[list[Tensor], dict[str, Tensor]]:
        """Adjoint of `_pair_curvatures`, returns `d (params, inputs)` and `d critical eigenvectors`: one pair and fn at a time"""
        n_fns, k = d_curvatures.shape[:2]
        rows, cols = torch.triu_indices(k, k)
        vjp_fn = COMPILED[curvature_grad_vjp] if self.compile else curvature_grad_vjp
        def symmetrize(d: Tensor, i: int, j: int) -> Tensor:
            """curvatures[i, j] == curvatures[j, i], so we need to add their gradients"""
            return d[:, i, j] if i == j else d[:, i, j] + d[:, j, i]

        results: list[tuple[list[Tensor], list[Tensor]]] = []  # per device: (d (params, inputs), d critical)
        for d, items in self._dealt(len(rows), len(losses)):
            loss = losses[d]
            U = {name: u.to(loss.device) for name, u in critical.items()}
            d_inputs = [torch.zeros(n_fns, *x.shape, dtype=x.dtype, device=loss.device) for x in (*loss.params.values(), *loss.inputs.values())]
            d_critical = [torch.zeros(n_fns, *u.shape, dtype=u.dtype, device=loss.device) for u in U.values()]
            for n in items:
                i, j = int(rows[n]), int(cols[n])
                d_curvature = symmetrize(d_curvatures, i, j).to(loss.device)
                d_curvature_grad = {name: symmetrize(d_C_grad, i, j).to(loss.device) for name, d_C_grad in d_curvature_grads.items()}
                for fn in range(n_fns):
                    d_params, d_loss_inputs, d_u, d_v = vjp_fn(
                        *loss, {name: u[i] for name, u in U.items()}, {name: u[j] for name, u in U.items()},
                        d_curvature[fn], {name: d_C_grad[fn] for name, d_C_grad in d_curvature_grad.items()},
                    )
                    for total, d_pair in zip(d_inputs, (*d_params.values(), *d_loss_inputs.values())):
                        total[fn] += d_pair
                    for total, d_first, d_second in zip(d_critical, d_u.values(), d_v.values()):
                        total[fn, i] += d_first
                        total[fn, j] += d_second
            results.append((d_inputs, d_critical))

        device = losses[0].device
        return (  # sum gradients over all pairs
            [torch.stack([d_x[n].to(device) for d_x, _ in results]).sum(0) for n in range(len(results[0][0]))],
            {name: torch.stack([d_u[n].to(device) for _, d_u in results]).sum(0) for n, name in enumerate(critical)},
        )

    def _eigvecs_vjp(
        self, hvp_fn: HVP, loss: Loss, critical: dict[str, Tensor], eigvals: Tensor, d_critical: dict[str, Tensor],
    ) -> list[Tensor]:
        """Adjoint of `_top_eigs` by implicit differentiation, returns `d (params, inputs)`:
            HU = lam U  ->  (lam - H) dU = dH U
            ->  v^T dU/dx = [(lam - H)^-1 v] dH U
        The IVHP is MINRES + projection, then `sum_i d(z_i^T H u_i)` (`curvature_grad`, one eigenvector and fn at a time)"""
        shapes = {name: p.shape for name, p in self.params.items()}
        n_fns, k = next(iter(d_critical.values())).shape[:2]
        U = flatten(critical)  # [#params, k]
        off_span = lambda V: V - U @ (U.T @ V)
        minus_H = lambda V: -off_span(flatten(hvp_fn(unflatten(off_span(V), shapes))))
        Y = off_span(flatten({name: d.flatten(0, 1) for name, d in d_critical.items()}))  # [#params, #fns * k]
        Z = off_span(minres(minus_H, Y, shift=eigvals.repeat(n_fns), rtol=self.VJP_RTOL)[0])
        Z = {
            name: z.reshape(n_fns, k, *p.shape)
            for name, (p, z) in dict_zip(self.params, unflatten(Z, shapes)).items()
        }
        grad_fn = COMPILED[curvature_grad] if self.compile else curvature_grad
        d_inputs = [torch.zeros(n_fns, *x.shape, dtype=x.dtype, device=x.device) for x in (*loss.params.values(), *loss.inputs.values())]
        for fn in range(n_fns):
            for i in range(k):
                (d_params, d_loss_inputs), _ = grad_fn(
                    *loss, {name: z[fn, i] for name, z in Z.items()}, {name: u[i] for name, u in critical.items()}
                )
                for total, d_pair in zip(d_inputs, (*d_params.values(), *d_loss_inputs.values())):
                    total[fn] += d_pair
        return d_inputs

    def vjp(
        self, t: int, forward_fn: Forward, inputs: Sequence[Tensor],
        cotangents: OptimState, next_state: OptimState | None = None,
    ) -> Sequence[Tensor]:
        """Backward of `step`, re-using the critical eigenvectors and curvatures from `next_state` (required).
        Runs in stages: first the adjoint of `_flow_update`, then of `_pair_curvatures`, then `_eigvecs_vjp`.
        `inputs`: the params, then keyword tensors of `forward_fn` (the example weights)"""
        assert isinstance(next_state, CentralFlowState), "the central flow backward needs the state `step` returned"
        grads = torch.autograd.grad(forward_fn(self.params)[0], self.params, create_graph=True)
        if next_state.n_critical == 0:
            params = self._gradient_flow_step(t, grads)
            return torch.autograd.grad(
                list(params.values()), inputs, list(cotangents), is_grads_batched=True
            )

        curvatures = next_state.curvatures["curvatures"].detach().requires_grad_()
        curvature_grads = {name: C_grad.detach().requires_grad_() for name, C_grad in next_state.curvature_grads.items()}
        params = self._flow_update(t, grads, curvatures, curvature_grads)[0]
        d_leaves = torch.autograd.grad(
            list(params.values()), [*inputs, curvatures, *curvature_grads.values()], list(cotangents),
            is_grads_batched=True, materialize_grads=True,
        )
        d_inputs, d_curvatures, d_curvature_grads = (
            d_leaves[: len(inputs)], d_leaves[len(inputs)], d_leaves[len(inputs) + 1 :]
        )
        del params, d_leaves, grads

        critical = next_state.critical
        with torch.no_grad():  # (torch.func from here on)
            losses = self._losses(forward_fn, inputs[len(self.params) :])
            d_pairs, d_critical = self._pair_curvatures_vjp(
                losses, critical, d_curvatures, dict(zip(critical, d_curvature_grads))
            )
            hvp_fn = partial(self._chunked_hvp, self._hvp_fns(forward_fn))
            d_eigvecs = self._eigvecs_vjp(hvp_fn, losses[0], critical, curvatures.detach().diagonal(), d_critical)
        return [d + d_p + d_e for d, d_p, d_e in zip(d_inputs, d_pairs, d_eigvecs)]

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        return {k: self.lr(t) / self.SUBSTEPS for k in self.params}


@dataclass(kw_only=True)
class NewtonState(SecondOrderState):
    """Damped Newton: `params <- params - lr * (H + shift I)^-1 (grads + wd * params)`.
    We set `shift = max(-2 min_eig, 0) + damping` so that the IVHP is well-conditioned.
    The backward pass (vjp) is custom and equivalent in cost to the forward pass (step)"""

    aux = ("newton", "eigvec", "stats")
    newton: dict[str, Tensor] = field(default_factory=dict)  # last IHVP solution (warm start)
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # most negative eig (warm start)

    damping: float = 0.1
    rtol: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not self.newton:
            self.newton = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            self.eigvec = {
                k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if not self.stats:
            p = next(iter(self.params.values()))
            nan = torch.tensor(float('nan'), dtype=p.dtype, device=p.device)
            self.stats = {
                "newton_iters": nan, "residual": nan, "min_eig": nan, "eig_iters": nan
            }

    def _solve(
        self, hvp_fn: HVP, shift: Tensor, rhs: dict[str, Tensor], x0: dict[str, Tensor] | None = None
    ) -> tuple[dict[str, Tensor], Tensor, int]:
        """Solve `(H + shift I)^-1 rhs` by MINRES with warm start `x0`"""
        shapes = {k: p.shape for k, p in self.params.items()}
        batched = next(iter(rhs.values())).ndim > len(next(iter(shapes.values())))
        H = lambda V: flatten(hvp_fn(unflatten(V, shapes)))  # H on flat columns [#params, m]
        X, residual, n_iters = minres(
            H, flatten(rhs, batched).detach(), shift=shift.detach(),
            x0=None if x0 is None else flatten(x0, batched), rtol=self.rtol
        )
        return unflatten(X, shapes), residual, n_iters

    def _shift(self, min_eig: Tensor) -> Tensor:
        return (-2 * min_eig).clamp_min(0) + self.damping

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        hvp_fn = hvp_from_grads(grads, self.params)
        min_eig, eigvec, eig_iters = hessian_eigs(hvp_fn, self.eigvec, sign=-1, batched=False)
        shift = self._shift(min_eig)

        reg_grads = {k: g + self.wd * p for k, (p, g) in dict_zip(self.params, grads).items()}
        newton, res, n_iters = self._solve(hvp_fn, shift, reg_grads, x0=self.newton)
        params = {k: p - self.lr(t) * x for k, (p, x) in dict_zip(self.params, newton).items()}

        stats = {
            "newton_iters": torch.full_like(res, n_iters), "residual": res,
            "min_eig": min_eig, "eig_iters": torch.full_like(min_eig, eig_iters),
        }
        return replace(self, params=params, newton=newton, eigvec=eigvec, stats=stats)

    def vjp(
        self, t: int, forward_fn: Forward, inputs: Sequence[Tensor], cotangents: OptimState, next_state: OptimState | None = None
    ) -> Sequence[Tensor]:
        """Implicit differentiation of a Newton step via one more IVHP.
        With `F = grads - (H + shift I) @ newton` and `adjoint = (H + shift I)^-1 @ cotangent`,
            `cotangent^T d(params - lr * newton) = cotangent^T d(params) - lr * adjoint^T dF`
        """
        grads = torch.autograd.grad(forward_fn(self.params)[0], self.params, create_graph=True)
        stepped = self._update(t, grads)
        newton, eigvec = stepped.newton, stepped.eigvec
        hvp_fn = hvp_from_grads(grads, self.params)
        reg_grads = {k: g + self.wd * p for k, (p, g) in dict_zip(self.params, grads).items()}

        H_eigvec, H_newton = hvp_fn(eigvec, create_graph=True), hvp_fn(newton, create_graph=True)
        min_eig = torch.stack([(v * Hv).sum() for v, Hv in dict_zip(eigvec, H_eigvec).values()]).sum()
        shift = self._shift(min_eig)  # with its graph
        F = {k: r - Hx - shift * x for k, (r, Hx, x) in dict_zip(reg_grads, H_newton, newton).items()}

        adjoint, _, _ = self._solve(hvp_fn, shift, cotangents.params)
        implicit = torch.autograd.grad(
            list(F.values()), inputs, list(adjoint.values()), is_grads_batched=True
        )
        direct = torch.autograd.grad(
            list(self), inputs, list(cotangents), is_grads_batched=True, materialize_grads=True
        )
        return [d - self.lr(t) * i for (d, i) in zip(direct, implicit)]

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """The step size in low curvature directions (where Newton ~ SGD)"""
        shift = self._shift(self.stats["min_eig"])
        return {k: self.lr(t) / shift for k in self.params}

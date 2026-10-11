import math
from dataclasses import dataclass, replace
from typing import Callable, ClassVar, Iterator, Self

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call
from tqdm import tqdm

from src.data import EvalLoader, Loader
from src.lobpcg import top_hessian_eigs
from src.utils import Logger, dict_zip, op_norm


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Mean over the classes of the squared error to the one-hot labels."""
    return (out - F.one_hot(Y, out.shape[1]).to(out.dtype)).pow(2).mean(1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


def metrics(out: Tensor, Y: Tensor) -> dict[str, Tensor]:
    """Per-example classification metrics of the model output, for logging: accuracy and margin."""
    correct = out.gather(1, Y[:, None])[:, 0]
    best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == Y).to(out.dtype), "margin": correct - best_other}


def eval_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], loader: EvalLoader, grad: bool = False
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss over the examples of `loader`, accumulated batch by batch to bound the memory.
    Returns: (loss, {metric: mean value}, {param: d loss / d param} if `grad` else {})."""
    params = {k: p.detach().requires_grad_(grad) for k, p in params.items()}
    loss, mean_metrics = torch.zeros((), device=loader.device), {}
    grads = {k: torch.zeros_like(p) for k, p in params.items()} if grad else {}
    for X, Y in loader:
        out = functional_call(model, params, (X,))
        term = loss_fn(out, Y).sum() / len(loader)
        if grad:
            dgrads = torch.autograd.grad(term, params)
            grads = {k: grads[k] + dgrads[k] for k in grads}
        loss += term.detach()
        for k, v in metrics(out.detach(), Y).items():
            mean_metrics[k] = mean_metrics.get(k, 0) + v.sum() / len(loader)
    return loss, mean_metrics, grads


@dataclass(kw_only=True)
class OptimState:
    """All model + optimizer state tensors: the dict fields named in `groups`."""

    groups: ClassVar[tuple[str, ...]] = ("params",)
    params: dict[str, Tensor]  # model parameters

    @property
    def param_groups(self) -> dict[str, dict[str, Tensor]]:
        return {group: getattr(self, group) for group in self.groups}

    def __iter__(self) -> Iterator[Tensor]:
        return (t for group in self.param_groups.values() for t in group.values())

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply `fn` to each state tensor. Returns a new OptimState."""
        new_fields = {
            group_name: {name: fn(tensor) for name, tensor in group.items()}
            for group_name, group in self.param_groups.items()
        }
        return replace(self, **new_fields)

    def unflatten(self, tensors: list[Tensor]) -> Self:
        """Inverse of iterating: a state like this one holding `tensors` instead."""
        it = iter(tensors)
        return self.apply(lambda _: next(it))

    def step(self, grads: dict[str, Tensor]) -> Self:
        """Update state tensors. Returns a new OptimState."""
        raise NotImplementedError

    @property
    def precond(self) -> dict[str, Tensor] | None:
        """Per-element multiplier of the gradient in the update, up to `lr` (None: 1). The Hessian
        eigenvalues to compare with `stability_threshold` are those of the preconditioned Hessian."""
        return None

    @property
    def stability_threshold(self) -> float:
        """Top (preconditioned) Hessian eigenvalue above which the step is unstable (edge of stability)."""
        raise NotImplementedError


@dataclass(kw_only=True)
class SGDState(OptimState):
    lr: float
    weight_decay: float = 0.0

    def step(self, grads: dict[str, Tensor]) -> Self:
        new_params = {
            name: p - self.lr * (g + self.weight_decay * p)
            for name, (p, g) in dict_zip(self.params, grads).items()
        }
        return replace(self, params=new_params)

    @property
    def stability_threshold(self) -> float:
        return 2 / self.lr


@dataclass(kw_only=True)
class SGDMomentumState(SGDState):
    groups = ("params", "buffer")
    momentum: float = 0.9
    buffer: dict[str, Tensor] | None = None  # None -> zeros

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def step(self, grads: dict[str, Tensor]) -> Self:
        assert self.buffer is not None
        new_buffer = {
            name: self.momentum * buf + g + self.weight_decay * p
            for name, (p, buf, g) in dict_zip(self.params, self.buffer, grads).items()
        }
        new_params = {
            name: p - self.lr * buf
            for name, (p, buf) in dict_zip(self.params, new_buffer).items()
        }
        return replace(self, params=new_params, buffer=new_buffer)

    @property
    def stability_threshold(self) -> float:
        return 2 * (1 + self.momentum) / self.lr


@dataclass(kw_only=True)
class AdamState(OptimState):
    """Adam as in torch.optim.Adam (L2 weight decay), with per-parameter learning rate
    multipliers `lr_mults` (None: 1; `models.adam_lr_mults` for μP)."""

    groups = ("params", "m", "v")
    lr: float
    lr_mults: dict[str, float] | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    t: int = 0  # steps taken
    m: dict[str, Tensor] | None = None  # None -> zeros
    v: dict[str, Tensor] | None = None

    def __post_init__(self):
        if self.lr_mults is None:
            self.lr_mults = {k: 1.0 for k in self.params}
        if self.m is None:
            self.m = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if self.v is None:
            self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def _step_sizes(self, v: dict[str, Tensor], t: int) -> dict[str, Tensor]:
        """Per-element multiplier of m in the update at step t, up to lr: mult / (bias-corrected sqrt(v) + eps)."""
        assert self.lr_mults is not None
        b1, b2 = self.betas
        return {k: self.lr_mults[k] / (1 - b1**t) / (v_k.sqrt() / (1 - b2**t) ** 0.5 + self.eps) for k, v_k in v.items()}

    def step(self, grads: dict[str, Tensor]) -> Self:
        assert self.m is not None and self.v is not None
        b1, b2 = self.betas
        t = self.t + 1
        grads = {k: g + self.weight_decay * p for k, (g, p) in dict_zip(grads, self.params).items()}
        new_m = {k: b1 * m + (1 - b1) * g for k, (m, g) in dict_zip(self.m, grads).items()}
        new_v = {k: b2 * v + (1 - b2) * g * g for k, (v, g) in dict_zip(self.v, grads).items()}
        new_params = {
            k: p - self.lr * size * m
            for k, (p, m, size) in dict_zip(self.params, new_m, self._step_sizes(new_v, t)).items()
        }
        return replace(self, params=new_params, m=new_m, v=new_v, t=t)

    @property
    def precond(self) -> dict[str, Tensor]:
        assert self.v is not None
        return self._step_sizes(self.v, max(self.t, 1))

    @property
    def stability_threshold(self) -> float:
        """Cohen et al. 2022, "Adaptive gradient methods at the edge of stability": 38/lr for beta1 = 0.9."""
        return 2 * (1 + self.betas[0]) / ((1 - self.betas[0]) * self.lr)


# ----- training -------------------------------------------------


class Magic:
    """Deterministic training of `model` on `loader` (`forward_pass`) and MAGIC (`backward_pass`): the gradient
    of eval losses after training w.r.t. the per-example weights of the training data, by backpropagating
    through the whole training.

    Logging settings (for `forward_pass(log=True)`; the backward pass logs whenever `logger` is given):
    - test_loader, eval_every: log the loss and metrics on `test_loader` every `eval_every` steps
    - num_top_eigs: log this many top (preconditioned) Hessian eigenvalues (0: none)
    - log_tangent: log the norms of the forward tangent ds_t/ds_0 v, for a random v with unit-norm tensors
    """

    def __init__(
        self,
        model: nn.Module,
        loader: Loader,
        loss_fn: LossFn,
        logger: Logger | None = None,
        *,
        test_loader: EvalLoader | None = None,
        eval_every: int = 10,
        num_top_eigs: int = 1,
        log_tangent: bool = True,
    ):
        self.model, self.loader, self.loss_fn, self.logger = model, loader, loss_fn, logger
        self.test_loader, self.eval_every = test_loader, eval_every
        self.n_eigs, self.log_tangent = num_top_eigs, log_tangent
        self.device, self.dataset = loader.device, loader.dataset

    def forward_pass(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        ckpt_every: int | None = None, log: bool = False,
    ) -> tuple[OptimState, list[OptimState]]:
        """Train on steps [start, start + n_steps).
        Returns: (final state, states before steps start, start + ckpt_every, ...), on the CPU.
        Logs at each step t (if `log`, which also shows a progress bar):
        - loss, acc, ...: on the batch, before the step
        - params/<param>, grads/<param>: `op_norm` of each parameter (before the step) and gradient
        - eig/i, lobpcg_iters: top Hessian eigenvalues (preconditioned if the state has a `precond`),
          normalized by the state's `stability_threshold` / 2 so that the edge of stability is at 2
        - tangent/<group>/<param>: norms of ds_{t+1}/ds_0 v
        - test loss, test acc, ...: on `test_loader`, after the step
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        ckpts: list[OptimState] = []
        logger = self.logger if log else None

        eigvecs: Tensor | None = None
        tangent: OptimState | None = None  # ds_t/ds_0 @ v
        if logger is not None and self.log_tangent:
            g = torch.Generator().manual_seed(start)
            tangent = state.apply(lambda t: torch.randn(t.shape, generator=g, dtype=t.dtype))
            tangent = tangent.apply(
                lambda t: (t / op_norm(t, n_iter=10)[0]).to(self.device, non_blocking=True)
            )

        for t in tqdm(range(start, start + n_steps), disable=logger is None):
            if ckpt_every and (t - start) % ckpt_every == 0:
                ckpts.append(state.apply(lambda t: t.to("cpu", copy=True)))

            X, Y, _, weights = self.loader.get(t)
            if tangent is not None:
                fwAD.enter_dual_level()
                state = state.unflatten([fwAD.make_dual(s, d) for s, d in zip(state, tangent)])
            params = {name: p.requires_grad_() for name, p in state.params.items()}

            out: Tensor = functional_call(self.model, params, (X,))
            loss = (weights * self.loss_fn(out, Y)).mean()
            grads = torch.autograd.grad(loss, params, create_graph=logger is not None and self.n_eigs > 0)
            state = state.step(grads)

            if tangent is not None:
                tangent = state.apply(lambda s: fwAD.unpack_dual(s).tangent.detach())
                fwAD.exit_dual_level()
            state = state.apply(lambda t: t.detach())

            if logger is not None:
                train_metrics = metrics(out.detach(), Y)
                logger.log(t, loss=loss, **{k: v.mean() for k, v in train_metrics.items()})
                logger.log_norms(t, "params", params)
                logger.log_norms(t, "grads", grads)
                if tangent is not None:
                    logger.log_norms(t, "tangent", tangent)
                if self.n_eigs > 0:
                    eigvals, eigvecs, n_iter = top_hessian_eigs(
                        grads, params, self.n_eigs, eigvecs, precond=state.precond, seed=start
                    )
                    eigvals = eigvals * (2 / state.stability_threshold)
                    logger.log(
                        t, **{f"eig/{i + 1}": v for i, v in enumerate(eigvals.unbind())},
                        lobpcg_iters=torch.tensor(n_iter),
                    )
                if self.test_loader is not None and ((t - start + 1) % self.eval_every == 0 or t - start == n_steps - 1):
                    val_loss, val_metrics, _ = eval_loss(self.model, self.loss_fn, state.params, self.test_loader)
                    logger.log(t, **{"test loss": val_loss} | {f"test {k}": v for k, v in val_metrics.items()})

        return state.apply(lambda t: t.cpu()), ckpts

    def rewind(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, passes: int = 2
    ) -> Iterator[OptimState]:
        """Reconstruct the states [start, start + n_steps] in reverse order.
        Cost: `passes` trainings. Memory: `passes * n_steps^(1/passes)` states.
        """
        chunk = math.ceil(n_steps ** ((passes - 1) / passes))  # 1 if passes == 1
        final, ckpts = self.forward_pass(
            init_state, n_steps=n_steps, start=start, ckpt_every=chunk
        )
        yield final
        if passes == 1:
            yield from reversed(ckpts)
            return
        for i, ckpt in reversed(list(enumerate(ckpts))):
            t0 = start + i * chunk
            inner = self.rewind(
                ckpt, n_steps=min(chunk, start + n_steps - t0), start=t0, passes=passes - 1
            )
            next(inner)  # its final state, already yielded
            yield from inner

    def backward_pass(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        eval_loaders: dict[str, EvalLoader], passes: int = 2,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """MAGIC over the steps [start, start + n_steps) from `init_state`, for the eval functions
        f = "mean loss over the examples of each eval loader" (named by the dict keys).
        Returns: ({name: f(s_T)}, {name: df(s_T)/dw}).
        Logs at each step t: delta/<name>/<group>/<param>, the norms of df/ds_t.
        Cost: `1 + passes + #fns` trainings; memory: `passes * n_steps^(1/passes) + #fns` states.
        """
        if len(eval_loaders) == 0:
            return {}, {}
        states = self.rewind(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(states).apply(lambda t: t.to(self.device))  # 1x traning

        zero_state = final_state.apply(torch.zeros_like)
        outputs: dict[str, Tensor] = {}     # f(s_T)
        deltas: dict[str, OptimState] = {}  # df/ds_t
        betas: dict[str, Tensor] = {}       # df/dw
        for name, eval_loader in eval_loaders.items():
            f_value, _, f_grad = eval_loss(
                self.model, self.loss_fn, final_state.params, eval_loader, grad=True
            )
            outputs[name] = f_value
            deltas[name] = replace(zero_state, params=f_grad)
            betas[name] = torch.zeros(len(self.dataset), device=self.device)

        steps = reversed(range(start, start + n_steps))
        for t, state in zip(steps, tqdm(states, total=n_steps)):  # `passes` trainings
            X, Y, indices, weights = self.loader.get(t)
            weights = weights.requires_grad_()
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())

            out: Tensor = functional_call(self.model, state.params, (X,))  # 1x fwd
            loss = (weights * self.loss_fn(out, Y)).mean()
            grads = torch.autograd.grad(loss, state.params, create_graph=True)  # 1x bwd
            new_state = state.step(grads)

            for i, (name, delta) in enumerate(deltas.items()):
                # (df/ds_t, df/dw_t) = ds_{t+1} / d(s_t, w_t) @ df/ds_{t+1}
                *d_state, d_weights = torch.autograd.grad(  # 1x double-bwd per `delta`
                    list(new_state), [*state, weights], list(delta),
                    retain_graph=i < len(deltas) - 1,
                )
                deltas[name] = state.unflatten(d_state)
                betas[name].index_add_(0, indices, d_weights)
                if self.logger is not None:
                    self.logger.log_norms(t, f"delta/{name}", deltas[name])

        return outputs, betas

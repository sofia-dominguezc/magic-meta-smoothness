import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

import random
from copy import deepcopy
from functools import partial
from itertools import accumulate
from typing import Callable, Iterable
from dataclasses import dataclass, replace

from utils import DeviceLogger
from lobpcg import torch_lobpcg


def criterion(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    assert logits.shape == labels.shape
    loss = (logits - labels).pow(2)
    if weights.ndim == 0:
        return weights / weights.numel() * loss.mean()
    return torch.einsum("i,i...->...", weights / weights.numel(), loss).mean()


def calc_acc(logits: Tensor, labels: Tensor):
    assert logits.shape == labels.shape
    acc = (logits > 0) == (labels > 0)
    return acc.sum() / acc.numel() * 100


all_acc = []
def log_acc(acc: Tensor):
    all_acc.append(acc.item())

all_loss = []
def log_loss(loss: Tensor):
    all_loss.append(loss.item())

all_eigvals = []
def log_eigvals(eigvals: Tensor):
    all_eigvals.append(eigvals.detach().cpu())


# ----- helpers --------------------------------------------------


@dataclass(frozen=True)
class DeterministicLoader:
    dataset: Dataset[tuple[Tensor, Tensor]]
    batch_size: int = 32
    seed: int = 0
    weight_fn: Callable[[int], float] = lambda _: 1.

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def get(
        self, step: int, *, device: torch.device | str | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return random batch from the dataset (deterministic in `step`)"""
        rng = random.Random(self.seed + step)
        indices = [rng.randint(0, len(self) - 1) for _ in range(self.batch_size)]
        batch = [self.dataset[idx] for idx in indices]
        weights = [self.weight_fn(idx) for idx in indices]
        if indices:
            X, Y = torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])
        else:
            print(f"empty at step {step}")
            _x, _y = self.dataset[0]
            X, Y = _x[None][:0], _y[None][:0]
        return (
            X.pin_memory().to(device, non_blocking=True),
            Y.pin_memory().to(device, non_blocking=True),
            torch.tensor(indices, dtype=torch.long).pin_memory().to(device, non_blocking=True),
            torch.tensor(weights, dtype=torch.float32).pin_memory().to(device, non_blocking=True),
        )


@dataclass(kw_only=True, frozen=True)
class TrainSettings:
    loader: DeterministicLoader
    lr: float = 0.001
    device: torch.device | str | None = None
    start: int = 0
    n_steps: int = 10_000
    ckpt_interval: int | None = None
    num_top_eigs: int = 0

    def optim_step(self, name: str, param: Tensor, grad: Tensor) -> Tensor:
        return param - self.lr * grad


def model_to_device(model: nn.Module, *, device: str | torch.device | None) -> nn.Module:
    for param in model.parameters():
        param.data = param.data.pin_memory()
    for name, buf in model.named_buffers():
        model.register_buffer(name, buf.pin_memory())
    return model.to(device, non_blocking=True)


@partial(torch.func.grad, argnums=4, has_aux=True)
def model_backward(
    model_arch: nn.Module, x: Tensor, y: Tensor, w: Tensor, params: dict[str, Tensor], /
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Returns a dict ∂loss/∂param, and aux (loss, acc).
    Should work for batched or non-batched inputs.
    """
    logits = torch.func.functional_call(model_arch, params, x)
    loss = criterion(logits, y, w)
    return loss, (loss.detach(), calc_acc(logits, y))


# ----- main functions -------------------------------------------


@torch.no_grad
def train_model(
    model: nn.Module, settings: TrainSettings, logger: DeviceLogger | None = None
) -> tuple[nn.Module, list[nn.Module]]:
    """Train model (determinisitcally) on steps [start, start + n_steps)
    and save checkpoints every `ckpt_interval` steps.
    """
    model = model_to_device(deepcopy(model), device=settings.device)
    param_offsets = list(accumulate(p.numel() for p in model.parameters()))
    init_eigs: Tensor | None = None
    ckpts: list[nn.Module] = []
    for step in range(settings.start, settings.start + settings.n_steps):
        if settings.ckpt_interval is not None and step % settings.ckpt_interval == 0:
            ckpts.append(deepcopy(model).cpu())

        X, Y, _, weights = settings.loader.get(step, device=settings.device)
        param_grads: dict[str, Tensor]; loss: Tensor; acc: Tensor
        param_grads, hvp_fn, (loss, acc) = torch.func.vjp(  # type: ignore
            partial(model_backward, model, X, Y, weights),
            dict(model.named_parameters()), has_aux=True,
        )
        if logger is not None:
            logger.log(loss=loss, acc=acc)
        if logger is not None and settings.num_top_eigs > 0:
            def JVP(flat_vector: Tensor) -> Tensor:
                dict_vector = {
                    name: flat_vector[offset - param.numel():offset].view_as(param)
                    for (name, param), offset in zip(model.named_parameters(), param_offsets)
                }
                dict_deriv, = hvp_fn(dict_vector)
                return torch.cat([v.flatten() for v in dict_deriv.values()])
            if init_eigs is None:
                init_eigs = torch.randn(  # assumes NN has `randn` initialization
                    param_offsets[-1], settings.num_top_eigs, device=settings.device
                )
            eigvals, init_eigs = torch_lobpcg(torch.func.vmap(JVP, 1, 1), init_eigs)
            logger.log(eigvals=eigvals)

        for (name, param), grad in zip(model.named_parameters(), param_grads.values()):
            param.data = settings.optim_step(name, param, grad)
    return model, ckpts


def reversed_model_states(
    ckpt_states: list[nn.Module], settings: TrainSettings
) -> Iterable[nn.Module]:
    """Given the checkpoints stored during training (every `sqrt(T)` steps),
    re-run training and return an iterable of all training steps from T-1 to 0.
    """
    assert len(ckpt_states)**2 == settings.n_steps, (
        f"{len(ckpt_states)}^2 != {settings.n_steps}"
    )
    ckpt_step = len(ckpt_states) - 1
    for model in reversed(ckpt_states):
        _, inner_ckpts = train_model(
            model,
            TrainSettings(
                loader=settings.loader,
                lr=settings.lr,
                device=settings.device,
                start=ckpt_step * len(ckpt_states),
                n_steps=len(ckpt_states),
                ckpt_interval=1,
                num_top_eigs=0,
            ),
        )
        for inner_model in reversed(inner_ckpts):
            yield model_to_device(inner_model, device=settings.device)
        ckpt_step -= 1


class GradPropagator:
    """Stores ∂f/∂s_t (current step) and a partial ∂f/∂w, and updates them"""

    def __init__(
        self, model: nn.Module, fn: Callable[[dict[str, Tensor]], Tensor], settings: TrainSettings
    ):
        self.deltas, self.result = torch.func.grad_and_value(fn)(  # ∂f/∂s_T, f(s_T)
            dict(model.named_parameters())
        )
        self.beta = torch.zeros(  # ∂f/∂w
            len(settings.loader), dtype=torch.float32, device=settings.device
        )
        self.optim_step = settings.optim_step

    def _weighted_step(
        self, params: dict[str, Tensor], batched_grads: dict[str, Tensor], weights: Tensor
    ) -> dict[str, Tensor]:
        w = weights / weights.numel()
        return {
            name: self.optim_step(name, param, torch.einsum("i,i...->...", w, b_grad))
            for (name, param), b_grad in zip(params.items(), batched_grads.values())
        }

    def step(
        self,
        params: dict[str, Tensor],
        batched_grads: dict[str, Tensor],
        idxs: Tensor,
        weights: Tensor,
        batch_hvp_fn: Callable[[dict[str, Tensor]], tuple[dict[str, Tensor]]],
    ):
        """Update `delta` and `beta` with the chain rule:
            ∂f/∂s_t = ∇_{s_t} (step_t(s_t, g_t, w_t) @ ∂f/∂s_{t+1})
            ∂f/∂w  += ∇_w (step_t(s_t, g_t, w_t) @ ∂f/∂s_{t+1})
        The gradient is split into the `s_t -> g_t` (hvp) and the `step_t` parts.
        """
        d_params, d_batched_grads, d_weights = torch.func.vjp(
            self._weighted_step, params, batched_grads, weights
        )[1](self.deltas)
        self.beta.scatter_add_(0, idxs, d_weights)
        d_hvp, = batch_hvp_fn(d_batched_grads)
        self.deltas = {name: d_params[name] + d_hvp[name] for name in self.deltas}


@torch.no_grad
def magic(
    model: nn.Module,
    ckpt_states: list[nn.Module],
    fns: list[Callable[[dict[str, Tensor]], Tensor]],
    settings: TrainSettings,
) -> list[tuple[Tensor, Tensor]]:
    """Compute a model's influence function of each `fn`.
    Returns list of tuples `f(s_T), ∂f/∂w` for each `f`

    The computational cost is `3 + len(fns)` model trainings, and the
    max memory usage is `O(sqrt(n_steps) + len(fns))` model states.
    """
    tangent_states = [GradPropagator(model, fn, settings) for fn in fns]
    step = settings.n_steps - 1
    for model_t in reversed_model_states(ckpt_states, settings):  # 1x training
        X, Y, idxs, weights = settings.loader.get(step, device=settings.device)
        params: dict[str, Tensor] = dict(model_t.named_parameters())
        batched_grads, batch_hvp_fn, _ = torch.func.vjp(  # type: ignore  # 1x training
            partial(
                torch.func.vmap(model_backward, (None, 0, 0, 0, None)),
                model_t, X, Y, weights,
            ),
            params, has_aux=True,
        )
        for state in tangent_states:  # 1x training per `fn`
            state.step(params, batched_grads, idxs, weights, batch_hvp_fn)
        step -= 1
    return [(state.result, state.beta) for state in tangent_states]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    device = 'cuda'
    DATA_SIZE = 512
    N_STEPS = 20**2
    DIM = 32

    class MajoritySignDataset(Dataset):
        def __init__(self, size: int):
            torch.manual_seed(0)
            self.X = torch.randn(size, DIM)
            self.Y = torch.where(self.X.sum(1, True) > 0, 1., -1.)
            self.Y = torch.where(torch.randn_like(self.Y) > 2, torch.randn_like(self.Y).sign(), self.Y)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, i: int):
            return self.X[i], self.Y[i]

    train_dataset = MajoritySignDataset(DATA_SIZE)

    loader = DeterministicLoader(train_dataset, batch_size=32, seed=42)
    model = nn.Sequential(
        nn.Linear(DIM, DIM),
        nn.GELU(),
        nn.Linear(DIM, 1),
    )
    settings = TrainSettings(
        loader=loader, lr=0.01, n_steps=N_STEPS, device=device
    )

    # print("1. Running forward training...")
    # model, ckpt_states = train_model(model, replace(settings, num_top_eigs=2))
    # print(f"   Collected {len(ckpt_states)} checkpoints")

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from itertools import accumulate
    # fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # for ax, data, name in zip(
    #     axes, [all_loss, all_acc, all_eigvals], ["Loss", "Accuracy", "Top eigenvalues"]
    # ):
    #     arr = np.asarray(data)
    #     alpha = 1.0 - np.exp(-np.log(2.0) / 10)
    #     ema = list(
    #         accumulate(arr, lambda prev, curr: alpha * curr + (1 - alpha) * prev)
    #     )
    #     ax.plot(range(len(data)), data, alpha=0.6, label="Raw")
    #     ax.plot(ema, label="EMA")
    #     ax.set(title=name, xlabel="Steps")
    #     ax.legend()
    # plt.tight_layout()
    # plt.show()
    # assert False

    def eval_fn(params: dict[str, Tensor], seed=0) -> Tensor:
        torch.manual_seed(seed + 1)
        X = torch.randn(1, DIM, device=device)
        return torch.func.functional_call(model, params, X).sum()

    fns = [partial(eval_fn, seed=seed) for seed in range(8)]
    final_model, linearization = magic(model, fns, settings)

    weight = torch.ones(len(train_dataset), device=device)
    all_preds = [[] for _ in fns]
    all_real = [[] for _ in fns]
    for i in tqdm(range(50)):
        torch.manual_seed(i + 10)
        idxss = torch.randint(0, len(weight) - 1, (DATA_SIZE // 20,))
        weight.fill_(1); weight[idxss] = 0  # remove random 5%
        shifted_model, _ = train_model(
            model,
            replace(
                settings,
                loader=replace(
                    settings.loader,
                    weight_fn=lambda k: 1. if k not in idxss.tolist() else 0.,
                ),
            ),
        )
        for (j, fn), (output, beta) in zip(enumerate(fns), linearization):
            predicted = output + beta @ (weight - 1)
            with torch.no_grad():
                real = fn(dict(shifted_model.named_parameters()))
            all_preds[j].append(predicted.item())
            all_real[j].append(real.item())
    all_ouputs = [output.item() for output, _ in linearization]

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    def plot_regression_grid(all_real: list[list[float]], all_preds: list[list[float]]):
        n_plots = len(all_real)
        ncols = min(n_plots, 4)
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, output in enumerate(all_ouputs):
            ax = axes_flat[idx]
            x = np.array(all_preds[idx])
            y = np.array(all_real[idx])
            n = len(x)

            # Linear regression
            res = stats.linregress(x, y)
            m, b = res.slope, res.intercept
            r2 = res.rvalue ** 2

            t_val = stats.t.ppf(0.975, df=n - 2)
            m_err = t_val * res.stderr

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = m * x_line + b

            # Residual standard error (estimate of sigma)
            y_pred = m * x + b
            s_err = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))

            # Variance component due to parameter estimation
            fit_var = 1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2)

            # 95% Confidence Interval (mean response line)
            se_line = s_err * np.sqrt(fit_var)
            ci_line = t_val * se_line

            # 95% Prediction Interval (individual noisy observations)
            se_pred = s_err * np.sqrt(1 + fit_var)
            pi_line = t_val * se_pred

            # Scatter data
            ax.scatter(x, y, alpha=0.6, edgecolors='none', label='Data')

            # Best fit line
            ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Best Fit')
            ax.plot(x_line, x_line, color='black', linestyle='--', linewidth=2, label='Reference')
            ax.scatter(output, output, edgecolors='none', label='Output', color='black')

            # 95% CI (line fit uncertainty)
            ax.fill_between(x_line, y_line - ci_line, y_line + ci_line, color='red', alpha=0.3, label='95% CI (Line)')

            # 95% PI (true noisy region for observations)
            ax.fill_between(x_line, y_line - pi_line, y_line + pi_line, color='gray', alpha=0.15, label='95% PI (Obs)')

            ax.set_xlabel('Pred')
            ax.set_ylabel('Real')
            ax.set_title(f'Plot {idx + 1}: Pred vs Real')

            ax.text(0.05, 0.78, f'$r^2$ = {r2:.3f}\nslope = {m:.3f} ± {m_err:.3f}', 
                    transform=ax.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.legend(loc='lower right', fontsize=8)

        # Remove unused grid axes if N is not a multiple of ncols
        for idx in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        plt.tight_layout()
        plt.show()

    plot_regression_grid(all_real, all_preds)

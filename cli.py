import click
import torch
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict

from data import SubsetSpec, create_eval_fn, get_mnist_dataset
from models import ModelConfig, SimpleCNN
from utils import (
    DeviceLogger,
    load_json,
    plot_regression_eval,
    save_json,
    save_training_plots,
)
from training import DeterministicLoader, TrainSettings, magic, train_model


def parse_int_list(ctx: click.Context, param: click.Parameter, value: str | None) -> list[int | None]:
    if not value:
        return [None]
    res: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        res.append(None if item in ("none", "null", "global") else int(item))
    return res


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--out-dir", type=click.Path(), default="./experiments")
@click.option("--labels", type=str, default="0,1,2,none", callback=parse_int_list, help="Comma-separated labels or 'none'")
@click.option("--frac", type=float, default=0.1)
@click.option("--seed", type=int, default=0)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=0.01)
@click.option("--n-steps", type=int, default=2500)
@click.option("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
def train_subsets(
    out_dir: str,
    labels: list[int | None],
    frac: float,
    seed: int,
    batch_size: int,
    lr: float,
    n_steps: int,
    device: str,
) -> None:
    base_path = Path(out_dir) / "subsets"
    train_dataset = get_mnist_dataset(train=True)
    dev = torch.device(device)

    for i, label in enumerate(labels):
        spec = SubsetSpec(label=label, frac=frac, drop=True, seed=seed + i)
        run_dir = base_path / spec.id
        run_dir.mkdir(parents=True, exist_ok=True)

        subset, dropped_idxs = spec.apply(train_dataset)
        loader = DeterministicLoader(subset, batch_size=batch_size, seed=0)
        
        model_config = ModelConfig()
        model = SimpleCNN(model_config).to(dev)
        
        ckpt_interval = int(n_steps ** 0.5)
        settings = TrainSettings(loader=loader, lr=lr, n_steps=n_steps, device=dev, ckpt_interval=ckpt_interval)

        click.echo(f"[{i+1}/{len(labels)}] Training subset model: {spec.id}")
        logger = DeviceLogger()
        model, ckpt_states = train_model(model, settings)
        metrics = logger.flush()

        save_json(spec.to_dict(), run_dir / "subset_spec.json")
        save_json(model_config.to_dict(), run_dir / "model_config.json")
        
        torch.save(model.state_dict(), run_dir / "final_model.pt")
        torch.save([m.state_dict() for m in ckpt_states], run_dir / "ckpt_states.pt")
        torch.save(dropped_idxs, run_dir / "dropped_indices.pt")
        
        save_training_plots(metrics, run_dir / "plots" / "training.png")


@cli.command()
@click.option("--out-dir", type=click.Path(), default="./experiments")
@click.option("--eval-labels", type=str, default="0,1,2,none", callback=parse_int_list)
@click.option("--eval-frac", type=float, default=0.1)
@click.option("--eval-seed", type=int, default=100)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=0.01)
@click.option("--n-steps", type=int, default=2500)
@click.option("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
def run_magic(
    out_dir: str,
    eval_labels: list[int | None],
    eval_frac: float,
    eval_seed: int,  # do we need this?
    batch_size: int,
    lr: float,
    n_steps: int,
    device: str,
) -> None:
    magic_dir = Path(out_dir) / "magic_baseline"
    magic_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    train_data = get_mnist_dataset(train=True)  # TODO: dataset flag
    test_data = get_mnist_dataset(train=False)

    loader = DeterministicLoader(train_data, batch_size=batch_size, seed=0)
    model_config = ModelConfig()
    model = SimpleCNN(model_config).to(dev)

    eval_specs = [
        SubsetSpec(label=lbl, frac=eval_frac, drop=False, seed=eval_seed + idx)
        for idx, lbl in enumerate(eval_labels)
    ]
    fns = [create_eval_fn(test_data, spec, model, dev) for spec in eval_specs]

    settings = TrainSettings(loader=loader, lr=lr, n_steps=n_steps, device=dev)

    click.echo("Running magic gradient calculation...")
    # TODO: wrong syntax
    final_model, linearization = magic(model, fns, settings)

    base_outputs = torch.tensor([out.item() for out, _ in linearization])
    betas = torch.stack([beta.cpu() for _, beta in linearization])

    save_json(model_config.to_dict(), magic_dir / "model_config.json")
    save_json([asdict(spec) for spec in eval_specs], magic_dir / "eval_specs.json")

    torch.save(final_model.state_dict(), magic_dir / "full_model.pt")
    torch.save({"base_outputs": base_outputs, "betas": betas}, magic_dir / "linearization.pt")


@cli.command()
@click.option("--out-dir", type=click.Path(), default="./experiments")
@click.option("--ckpt-idx", type=int, default=-1, help="Checkpoint index to evaluate (-1 for final)")
@click.option("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
def eval_approx(out_dir: str, ckpt_idx: int, device: str) -> None:
    base_path = Path(out_dir)
    magic_dir = base_path / "magic_baseline"
    subsets_dir = base_path / "subsets"

    dev = torch.device(device)
    linearization = torch.load(magic_dir / "linearization.pt", weights_only=True)
    base_outputs = linearization["base_outputs"]
    betas = linearization["betas"]

    eval_specs_json = load_json(magic_dir / "eval_specs.json")
    eval_specs = [SubsetSpec(label=d["label"], frac=d["frac"], drop=False, seed=d["seed"]) for d in eval_specs_json]

    model_config = ModelConfig(**load_json(magic_dir / "model_config.json"))
    test_data = get_mnist_dataset(train=False)  # TODO: store dataset (type) in metadata

    subset_dirs = [d for d in subsets_dir.iterdir() if d.is_dir()]
    
    all_preds: list[list[float]] = [[] for _ in eval_specs]
    all_reals: list[list[float]] = [[] for _ in eval_specs]

    click.echo(f"Evaluating linear approximation on {len(subset_dirs)} retrained models...")
    for sdir in tqdm(subset_dirs):
        dropped_idxs = torch.load(sdir / "dropped_indices.pt", weights_only=True)
        
        weight_delta = torch.zeros(betas.size(1))
        weight_delta[dropped_idxs] = -1.0
        
        predicted = base_outputs + betas @ weight_delta

        model = SimpleCNN(model_config).to(dev)
        if ckpt_idx == -1:
            model.load_state_dict(torch.load(sdir / "final_model.pt", weights_only=True))
        else:
            ckpt_states = torch.load(sdir / "ckpt_states.pt", weights_only=True)
            model.load_state_dict(ckpt_states[ckpt_idx])

        fns = [create_eval_fn(test_data, spec, model, dev) for spec in eval_specs]
        real_vals = [fn(dict(model.named_parameters())).item() for fn in fns]

        for idx in range(len(eval_specs)):
            all_preds[idx].append(predicted[idx].item())
            all_reals[idx].append(real_vals[idx])

    eval_out = base_path / "evaluation"
    eval_out.mkdir(parents=True, exist_ok=True)

    plot_regression_eval(
        all_preds,
        all_reals,
        [spec.id for spec in eval_specs],
        eval_out / "pred_vs_real.png",
    )
    click.echo(f"Evaluation complete. Results saved to {eval_out}")


if __name__ == "__main__":
    cli()

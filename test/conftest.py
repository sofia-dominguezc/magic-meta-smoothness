import sys
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # no windows during tests
sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(autouse=True)
def restore_default_dtype():
    """`Experiment` sets the default dtype from its config: undo it after each test."""
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def float64():
    """Exact gradient comparisons need float64 (float32 round-off accumulates over the steps)."""
    torch.set_default_dtype(torch.float64)

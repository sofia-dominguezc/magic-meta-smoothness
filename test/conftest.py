import sys
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # no windows during tests
sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture
def float64():
    """Exact gradient comparisons need float64 (float32 round-off accumulates over the steps)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)

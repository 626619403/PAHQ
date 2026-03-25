"""Shared pytest fixtures for PAHQ unit tests.

All fixtures are CPU-compatible: they use tiny dimensions so that tests run
in CI without a GPU.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Tiny weight tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def d_model():
    return 8


@pytest.fixture
def n_heads():
    return 2


@pytest.fixture
def d_head(d_model, n_heads):
    return d_model // n_heads  # 4


@pytest.fixture
def weight_2d(d_model):
    """A small 2-D weight matrix [d_model, d_model]."""
    torch.manual_seed(0)
    return torch.randn(d_model, d_model)


@pytest.fixture
def weight_batch(d_model):
    """A batch of 4 weight vectors [4, d_model]."""
    torch.manual_seed(1)
    return torch.randn(4, d_model)


# ---------------------------------------------------------------------------
# Tiny correspondence / graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_edge_scores():
    """Minimal dict of edge scores for threshold-sweep tests.

    Keys are simplified strings rather than real TorchIndex objects so that
    tests can validate threshold logic independently of ACDC internals.
    """
    return {
        "edge_A": 0.5,
        "edge_B": 1.2,
        "edge_C": 0.1,
        "edge_D": 2.0,
        "edge_E": 0.0,
    }

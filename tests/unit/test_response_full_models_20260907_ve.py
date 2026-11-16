from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.response_full_models_20260907_span import projected_trace
from experiments.response_full_models_20260907_ve import canonical_posterior, shared_queries


def test_edm_posterior_exact_trace_includes_skip_and_both_native_factors():
    class Linear(torch.nn.Module):
        def forward(self, x, scale):
            return 2*x + 3
    result = SimpleNamespace(model=Linear(), config=SimpleNamespace(
        diffusion_preconditioning="edm", diffusion_data_std=1.5))
    x = torch.randn(5, 7, dtype=torch.float64)
    s = .37
    denominator = s*s+1.5**2
    gain = 1.5**2/denominator + 2*s*1.5/denominator
    intercept = 3*s*1.5/denominator**.5
    value, trace = projected_trace(lambda q: canonical_posterior(result, q, s), x, torch.eye(7), 2)
    torch.testing.assert_close(value, gain*x+intercept)
    torch.testing.assert_close(trace, torch.full((5,), 7*gain, dtype=x.dtype))


def test_shared_queries_verify_raw_rows_and_preserve_ids_across_datasets():
    test = np.arange(60, dtype=np.float32).reshape(10, 6)
    fixture = {"dataset": "spiral", "query_ids": np.array([7, 2, 9]), "query_raw": test[[7, 2, 9]]}
    ids, raw = shared_queries(fixture, "spiral", test, 2)
    np.testing.assert_array_equal(ids, [7, 2])
    np.testing.assert_array_equal(raw, test[[7, 2]])
    _, other = shared_queries(fixture, "exp", test+1, None)
    np.testing.assert_array_equal(other, test[[7, 2, 9]]+1)
    with pytest.raises(AssertionError):
        shared_queries(fixture, "spiral", test+1, None)


def test_shared_queries_rejects_duplicate_or_out_of_range_ids():
    test = np.ones((5, 3), dtype=np.float32)
    for ids in [[1, 1], [-1, 2], [1, 5]]:
        with pytest.raises(ValueError):
            shared_queries({"dataset": "other", "query_ids": ids}, "this", test, None)

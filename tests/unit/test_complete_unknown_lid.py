import numpy as np
import pytest

from experiments import fair_measurements as measurement
from experiments.fair_campaign import inventory
from experiments.fair_protocol import geometry, resolve
from scripts.complete_unknown_lid import evaluation_grid
from scripts.run_fair_cells import reference_key


def _config(variant):
    geo = geometry('e1_spiral_pca', 'coefficients', (2,))
    config, _ = resolve(variant, geo, rank=geo['ambient_dim'], device='cpu')
    return config


def test_extended_grid_only_for_unbounded_support():
    grid = evaluation_grid(_config('t_flowmatching'), 999.)
    assert len(grid) == 29 and grid[0] == 1 / 256 and grid[-1] == 999.
    assert np.all(np.diff(np.log(grid)) > 0)
    with pytest.raises(ValueError, match='unbounded'):
        evaluation_grid(_config('rectified_flow'), 999.)


def test_default_grid_is_the_campaign_grid():
    config = _config('scale_conditioned_nf')
    assert np.array_equal(evaluation_grid(config, None), measurement.model_scales(config))


def test_runner_resolves_frozen_scale_references():
    cells = {c.key: c for c in inventory()[1]}
    assert reference_key(cells['e1/e1_sampled_fmnist_step1/dataset']) is None
    assert reference_key(cells['e1/e1_sampled_fmnist_step7/dataset']) == 'e1/e1_sampled_fmnist_step1/dataset'
    assert reference_key(cells['e5/e5_downscaled_fmnist/dataset']) is None
    assert reference_key(cells['e5/e5_stretched_power4/dataset']) == 'e5/e5_downscaled_fmnist/dataset'
    assert reference_key(cells['e1/e1_spiral_pca/coefficients']) is None

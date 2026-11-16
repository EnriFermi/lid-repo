import math

import pytest
import torch

from experiments.fm_mechanism_regional_transfer import observed_regions


def fixture():
    return {"data_law": "uniform", "latent_half_width": 2.,
            "tangent_basis": torch.tensor([[1.], [0.]], dtype=torch.float64)}


def test_observed_partition_is_disjoint_and_exhaustive_including_outside_support():
    points = torch.tensor([[0., 7.], [1.6, 0.], [1.9, -8.], [2.3, 0.]], dtype=torch.float64)
    parts = observed_regions(points, fixture(), .2)
    assert parts["interior"].tolist() == [True, False, False, False]
    assert parts["transition"].tolist() == [False, True, False, False]
    assert parts["boundary"].tolist() == [False, False, True, True]
    assert torch.equal(sum(p.int() for p in parts.values()), torch.ones(4, dtype=torch.int))


def test_partition_uses_observed_tangent_location_not_normal_displacement():
    points = torch.tensor([[0., 0.], [1.9, 0.]], dtype=torch.float64)
    shifted = points + torch.tensor([0., 100.], dtype=torch.float64)
    left, right = observed_regions(points, fixture(), .2), observed_regions(shifted, fixture(), .2)
    assert all(torch.equal(left[k], right[k]) for k in left)
    angle = .71
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    rotated_fixture = fixture()
    rotated_fixture["tangent_basis"] = rotation @ rotated_fixture["tangent_basis"]
    rotated = observed_regions(points @ rotation.T, rotated_fixture, .2)
    assert all(torch.equal(left[k], rotated[k]) for k in left)


def test_rejects_unimplemented_geometry():
    changed = fixture()
    changed["data_law"] = "gaussian"
    with pytest.raises(ValueError, match="Uniform"):
        observed_regions(torch.zeros(2, 2, dtype=torch.float64), changed, .2)

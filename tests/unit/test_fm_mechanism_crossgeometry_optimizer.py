from dataclasses import replace

import pytest
import torch

from experiments.fm_mechanism_crossgeometry_optimizer import run, sample_batch
from experiments.fm_mechanism_pilot import PilotConfig, train_pilot
from experiments.fm_mechanism_crossgeometry_analysis import temporal_energy


@pytest.mark.parametrize("data_law,dimension,intrinsic", [("uniform", 8, 2), ("circle", 2, 1)])
def test_crossgeometry_continuation_independent_states_and_unchanged_source(tmp_path, data_law, dimension, intrinsic):
    config = PilotConfig(ambient_dim=dimension, intrinsic_dim=intrinsic, hidden_dim=16,
        depth=1, batch_size=8, steps=2, validation_count=8, log_interval=2,
        checkpoints=(0, 2), data_law=data_law, device="cpu")
    train_pilot(tmp_path/"source", config)
    source = tmp_path/"source/checkpoint_step_000002.pt"
    original = torch.load(source, weights_only=False)
    result = run(source, tmp_path/"paired", device="cpu", steps=2, queries=4, interval=1)
    assert result["source_unchanged"] and result["all_optimizer_steps_correct"]
    a = torch.load(tmp_path/"paired/lr1e3/checkpoint_step_000004.pt", weights_only=False)
    b = torch.load(tmp_path/"paired/lr1e4/checkpoint_step_000004.pt", weights_only=False)
    assert torch.equal(a["training_generator_state"], b["training_generator_state"])
    for s in original["models"]:
        assert all(int(st["step"]) == 4 for st in a["optimizers"][s]["state"].values())
        assert all(int(st["step"]) == 4 for st in b["optimizers"][s]["state"].values())
    # The original-LR member must be bitwise identical to normal pilot resume.
    train_pilot(tmp_path/"source", replace(config, steps=4), resume=source)
    continued = torch.load(tmp_path/"source/checkpoint_step_000004.pt", weights_only=False)
    for s in original["models"]:
        assert all(torch.equal(v, continued["models"][s][k]) for k,v in a["models"][s].items())
    assert torch.equal(a["training_generator_state"], continued["training_generator_state"])
    averaged = torch.load(tmp_path/"paired/lr1e3_tailavg/checkpoint_step_000004.pt", weights_only=False)
    assert averaged["optimizers"] == {}
    assert averaged["probe_metadata"]["inference_only_no_optimizer_state"]


def test_temporal_decomposition_preserves_curved_normal_and_tangent_components():
    torch.manual_seed(41)
    angles = torch.linspace(0., 2., 7, dtype=torch.float64)
    query = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    tangent = torch.stack((-query[:, 1], query[:, 0]), -1)
    values = .9*query + .03*torch.randn(5, 7, 2, dtype=torch.float64)
    project = lambda x: (x*tangent).sum(-1, keepdim=True)
    result = temporal_energy(values, query, project, .3)
    assert result["energy_identity_residual"] < 1e-14
    assert result["temporal_variance_energy"] > result["tangent_temporal_variance_energy"] > 0
    assert result["normal_correction_of_mean_function"] > 0

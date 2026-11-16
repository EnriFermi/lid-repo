import copy
import math

import pytest
import torch

from experiments import fm_scheduler_cause_training as training
from experiments.fm_mechanism_pilot import PilotConfig, train_pilot
from experiments.fm_scheduler_cause_torus import (
    LAW, draw_batch, installed_adapters, make_fixture, parser, posterior,
    prepare_source, run, sample, sha256, tangent_energy,
)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def source_fixture(path):
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
        batch_size=8, steps=2, validation_count=16, log_interval=2,
        checkpoints=(0, 2), device="cpu", data_law="uniform")
    train_pilot(path, config)
    return path/"checkpoint_step_000000.pt"


def test_product_geometry_has_local_d_not_span_dimension_and_rms_one():
    fixture = make_fixture(784, 20)
    basis = fixture["embedding_basis"]
    assert basis.shape == (784, 40) and "tangent_basis" not in fixture
    torch.testing.assert_close(basis.T@basis, torch.eye(40, dtype=torch.float64), atol=2e-14, rtol=2e-14)
    clean, _ = sample(fixture, 17, 190, "cpu")
    torch.testing.assert_close(clean.square().sum(-1), torch.full((17,), 784., dtype=torch.float64))
    radii = (clean@basis).reshape(17, 20, 2).norm(dim=-1)
    torch.testing.assert_close(radii, torch.full_like(radii, math.sqrt(784/20)))
    eye = torch.eye(784, dtype=torch.float64)
    energy = tangent_energy(eye, clean[:1].expand(784, -1), fixture)
    torch.testing.assert_close(energy.sum(), torch.tensor(20., dtype=torch.float64))
    assert fixture["linear_span_dimension"] == 40


@pytest.mark.parametrize("scale", [.01, .316227766, 1.])
def test_posterior_trace_matches_autograd_and_finite_difference(scale):
    fixture = make_fixture(6, 2, seed=102)
    clean, noise = sample(fixture, 1, 100, "cpu")
    point = (clean+.15*noise)[0]
    truth = posterior(point, scale, fixture)
    jacobian = torch.autograd.functional.jacobian(lambda q: posterior(q, scale, fixture).mean, point)
    torch.testing.assert_close(jacobian.trace(), truth.response, rtol=2e-10, atol=2e-10)
    step = 1e-5
    eye = torch.eye(6, dtype=torch.float64)
    numerical = torch.stack([(posterior(point+step*e, scale, fixture).mean
                            -posterior(point-step*e, scale, fixture).mean)/(2*step) for e in eye], dim=1)
    torch.testing.assert_close(numerical, jacobian, rtol=3e-6, atol=1e-8)
    torch.testing.assert_close(truth.full, truth.response+truth.correction)


def test_tiny_scale_clean_oracle_converges_to_local_dimension_with_curvature():
    fixture = make_fixture(8, 2, seed=31)
    clean, _ = sample(fixture, 3, 40, "cpu")
    truth = posterior(clean, 1e-4, fixture)
    torch.testing.assert_close(truth.response, torch.full((3,), 2., dtype=torch.float64), rtol=1e-9, atol=1e-9)
    assert bool((truth.correction > 0).all())
    assert bool(torch.isfinite(truth.full).all())


def test_rotation_and_ambient_normal_invariance_include_normal_correction():
    fixture = make_fixture(7, 2, seed=41)
    clean, noise = sample(fixture, 4, 42, "cpu")
    scales = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    basis = fixture["embedding_basis"]
    normal = noise-(noise@basis)@basis.T
    reference = posterior(clean, scales, fixture)
    displaced = posterior(clean+normal, scales, fixture)
    torch.testing.assert_close(displaced.mean, reference.mean, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(displaced.response, reference.response, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(displaced.correction-reference.correction, normal.square().sum(-1)/scales.square())
    torch.testing.assert_close(tangent_energy(normal, clean, fixture), torch.zeros(4, dtype=torch.float64), atol=1e-25, rtol=0)
    generator = torch.Generator().manual_seed(44)
    rotation = torch.linalg.qr(torch.randn(7, 7, generator=generator, dtype=torch.float64)).Q
    rotated_fixture = {**fixture, "embedding_basis": rotation@basis}
    rotated = posterior((clean+normal)@rotation.T, scales, rotated_fixture)
    torch.testing.assert_close(rotated.mean, displaced.mean@rotation.T)
    torch.testing.assert_close(rotated.full, displaced.full)


def test_training_draw_is_exactly_paired_and_evaluation_does_not_advance_it():
    fixture = make_fixture(8, 2, seed=88)
    config = {"batch_size": 5, "ambient_dim": 8, "lambda_min": .01, "lambda_max": 1.}
    left = torch.Generator().manual_seed(89)
    right = torch.Generator().manual_seed(89)
    for _ in range(3):
        before = left.get_state().clone()
        sample(fixture, 4, 123, "cpu")
        assert torch.equal(before, left.get_state())
        first = draw_batch(config, fixture, left, "cpu")
        second = draw_batch(config, fixture, right, "cpu")
        for a, b in zip(first, second):
            assert torch.equal(a, b)
        assert torch.equal(left.get_state(), right.get_state())
        assert bool(((first[2] >= .01)&(first[2] <= 1.)).all())


def test_prepare_source_preserves_every_original_payload_field_except_law(tmp_path):
    source = source_fixture(tmp_path/"source")
    initial_hash = sha256(source)
    prepare_source(source, tmp_path/"torus")
    original = torch.load(source, weights_only=False)
    prepared = torch.load(tmp_path/"torus/checkpoint_step_000000.pt", weights_only=False)
    assert prepared["config"]["data_law"] == LAW
    assert prepared.pop("product_circle_provenance")["source_sha256"] == initial_hash
    prepared["config"]["data_law"] = original["config"]["data_law"]
    assert_tree_equal(prepared, original)
    assert sha256(source) == initial_hash
    with pytest.raises(FileExistsError):
        prepare_source(source, tmp_path/"torus")


def test_adapters_restore_on_failure_and_do_not_redirect_other_laws():
    original = training.sample
    with pytest.raises(RuntimeError):
        with installed_adapters():
            fixture = {"tangent_basis": torch.eye(3)[:, :1], "data_law": "gaussian", "latent_std": 1.}
            a = original(fixture, 2, 3, "cpu")
            b = training.sample(fixture, 2, 3, "cpu")
            for x, y in zip(a, b):
                assert torch.equal(x, y)
            raise RuntimeError("test restoration")
    assert training.sample is original


@pytest.mark.parametrize("normalization,coordinate", [
    ("layernorm", "data"), ("layernorm", "native"), ("none", "native")])
def test_tiny_wrapper_retains_scalar_loss_and_averaged_checkpoint_contract(tmp_path, normalization, coordinate):
    source = source_fixture(tmp_path/"source")
    prepare_source(source, tmp_path/"torus")
    cli = ["train", "--source", str(tmp_path/"torus/checkpoint_step_000000.pt"),
        "--output", str(tmp_path/"trained"), "--device", "cpu", "--steps", "2",
        "--decay-step", "1", "--tail-average", "1", "--tail-probes", "1",
        "--query-count", "16", "--heldout-count", "16", "--probe-count", "8", "--checkpoints", "0", "2"]
    cli += ["--output-normalization", normalization, "--output-coordinate", coordinate]
    args = parser().parse_args(cli)
    result = run(args)
    assert result["source_unchanged"]
    fixture = torch.load(tmp_path/"trained/data.pt", weights_only=False)
    original_loss = training.original_loss
    for arm in training.ARM_SCHEDULE:
        field, payload = training.load_cause_checkpoint(tmp_path/f"trained/{arm}_tailavg/checkpoint_step_000002.pt")
        assert payload["config"]["data_law"] == LAW and payload["averaged"]
        assert field.pilot_output_coordinate == coordinate
        assert isinstance(field.output[0], torch.nn.LayerNorm if normalization == "layernorm" else torch.nn.Identity)
        gen = torch.Generator().manual_seed(56)
        clean, noise, scales = draw_batch(payload["config"], fixture, gen, "cpu")
        pred = training.arm_posterior(field, clean+scales[:, None]*noise, scales, arm)
        direct = ((pred-clean).square().mean(-1)/scales.square()).mean()
        torch.testing.assert_close(original_loss(field, clean, noise, scales, arm), direct)
    assert (tmp_path/"trained/product_circle_manifest.json").is_file()

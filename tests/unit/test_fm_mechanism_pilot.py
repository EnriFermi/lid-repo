from dataclasses import replace
import copy

import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_pilot import (
    PilotConfig, SCHEDULES, canonical_posterior, gaussian_posterior,
    load_pilot_checkpoint, make_fixture, train_pilot, ColumnIsotropicAdamW, population_posterior,
    _new_fields,
)
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def test_flat_fixture_and_gaussian_oracle() -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, validation_count=32, device="cpu")
    fixture = make_fixture(config)
    q, n, x = fixture["tangent_basis"], fixture["normal_basis"], fixture["clean_queries"]
    torch.testing.assert_close(q.T @ q, torch.eye(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(x @ n, torch.zeros(32, 6), atol=1e-6, rtol=0)
    result = gaussian_posterior(x, 0.5, fixture)
    torch.testing.assert_close(result, x * (4.0 / 4.25), atol=1e-6, rtol=1e-6)


def test_no_output_normalization_preserves_common_initialization() -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16,
                         depth=1, device="cpu")
    baseline, architecture = _new_fields(config)
    rng_baseline = torch.get_rng_state().clone()
    intervention, changed_architecture = _new_fields(replace(config, output_normalization="none"))
    assert torch.equal(torch.get_rng_state(), rng_baseline)
    assert architecture == changed_architecture
    for schedule in SCHEDULES:
        original = baseline[schedule].state_dict()
        changed = intervention[schedule].state_dict()
        assert original.keys() - changed.keys() == {"output.0.weight", "output.0.bias"}
        assert all(torch.equal(original[key], value) for key, value in changed.items())
        assert isinstance(intervention[schedule].output[0], torch.nn.Identity)
        assert isinstance(intervention[schedule].output[1], torch.nn.SiLU)
        assert isinstance(intervention[schedule].output[2], torch.nn.Linear)


def test_no_output_normalization_checkpoint_resume_and_loader(tmp_path) -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
                         batch_size=8, steps=2, validation_count=8, log_interval=1,
                         checkpoints=(0, 1, 2), device="cpu", output_normalization="none")
    full = train_pilot(tmp_path / "full", config)
    train_pilot(tmp_path / "resume", replace(config, steps=1))
    resumed = train_pilot(tmp_path / "resume", config,
                          resume=tmp_path / "resume" / "checkpoint_step_000001.pt")
    assert full["final_state_hashes"] == resumed["final_state_hashes"]
    field, payload = load_pilot_checkpoint(tmp_path / "full" / "checkpoint_step_000002.pt", "log_noise")
    assert isinstance(field.output[0], torch.nn.Identity)
    assert payload["config"]["output_normalization"] == "none"


def test_canonical_input_collapses_schedule_forward_and_gradients() -> None:
    torch.manual_seed(11)
    model = ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=8, hidden_dim=16, depth=1, zero_init_output=False))
    x = torch.randn(5, 8, requires_grad=True)
    lam = torch.logspace(-2, 0, 5)
    values = [canonical_posterior(model, x, lam, schedule, input_coordinate="channel") for schedule in SCHEDULES]
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], atol=0, rtol=0)
    grads = [torch.autograd.grad(value.square().sum(), x, retain_graph=True)[0] for value in values]
    for gradient in grads[1:]:
        torch.testing.assert_close(gradient, grads[0], atol=0, rtol=0)


def test_canonical_input_schedule_equality_through_adamw_update() -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
                         input_coordinate="channel", output_coordinate="data", device="cpu")
    torch.manual_seed(27)
    # Nonzero output weights ensure this test exercises hidden gradients too.
    template = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=config.ambient_dim, hidden_dim=config.hidden_dim,
        depth=config.depth, zero_init_output=False)).double()
    fields = [copy.deepcopy(template) for _ in SCHEDULES]
    clean = torch.randn(7, config.ambient_dim, dtype=torch.float64)
    noise = torch.randn_like(clean)
    lam = torch.logspace(-2, 0, len(clean), dtype=torch.float64)
    points = clean + lam[:, None] * noise
    outputs, losses, gradients = [], [], []
    for schedule, field in zip(SCHEDULES, fields):
        optimizer = torch.optim.AdamW(field.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        output = canonical_posterior(field, points, lam, schedule,
                                     input_coordinate=config.input_coordinate,
                                     output_coordinate=config.output_coordinate)
        loss = ((output-clean).square().mean(-1)/lam.square()).mean()
        loss.backward()
        outputs.append(output.detach())
        losses.append(loss.detach())
        gradients.append([parameter.grad.detach().clone() for parameter in field.parameters()])
        torch.nn.utils.clip_grad_norm_(field.parameters(), config.gradient_clip)
        optimizer.step()
    for i in (1, 2):
        assert torch.equal(outputs[i], outputs[0])
        assert torch.equal(losses[i], losses[0])
        assert all(torch.equal(a, b) for a, b in zip(gradients[i], gradients[0]))
        assert all(torch.equal(fields[i].state_dict()[key], fields[0].state_dict()[key])
                   for key in fields[0].state_dict())


@pytest.mark.parametrize("head_preconditioner", ["elementwise", "isotropic_column"])
def test_resume_restores_matched_stream_and_optimizer(tmp_path, head_preconditioner) -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
                         batch_size=8, steps=4, validation_count=8, log_interval=2,
                         checkpoints=(0, 2, 4), device="cpu", head_preconditioner=head_preconditioner)
    full = train_pilot(tmp_path / "full", config)
    train_pilot(tmp_path / "resume", replace(config, steps=2))
    resumed = train_pilot(tmp_path / "resume", config,
                          resume=tmp_path / "resume" / "checkpoint_step_000002.pt")
    assert full["final_state_hashes"] == resumed["final_state_hashes"]
    field, payload = load_pilot_checkpoint(tmp_path / "full" / "checkpoint_step_000004.pt", "log_noise")
    assert payload["examples_seen_per_schedule"] == 32
    assert field.config.ambient_dim == 8
    initial = torch.load(tmp_path / "full" / "checkpoint_step_000000.pt", weights_only=False)
    for state in initial["models"].values():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, initial["models"]["log_noise"][name], atol=0, rtol=0)


def test_column_isotropic_head_preserves_tangent_span() -> None:
    torch.manual_seed(7)
    basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q
    tangent, normal = basis[:, :3], basis[:, 3:]
    weights = torch.nn.Parameter(torch.zeros(16, 7, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(16, dtype=torch.float64))
    optimizer = ColumnIsotropicAdamW([weights, bias], lr=0.01, weight_decay=0.01)
    for _ in range(10):
        weights.grad = tangent @ torch.randn(3, 7, dtype=torch.float64)
        bias.grad = tangent @ torch.randn(3, dtype=torch.float64)
        optimizer.step()
    torch.testing.assert_close(normal.T @ weights, torch.zeros(13, 7, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(normal.T @ bias, torch.zeros(13, dtype=torch.float64), atol=1e-15, rtol=0)


def test_native_output_lognoise_is_identical_and_loader_preserves_flag(tmp_path) -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
                         batch_size=8, steps=4, validation_count=8, log_interval=2,
                         checkpoints=(0, 4), device="cpu", schedules=("log_noise",))
    baseline = train_pilot(tmp_path / "data", config)
    intervention = train_pilot(tmp_path / "native", replace(config, output_coordinate="native"))
    assert baseline["final_state_hashes"] == intervention["final_state_hashes"]
    field, _ = load_pilot_checkpoint(tmp_path / "native" / "checkpoint_step_000004.pt", "log_noise")
    assert field.pilot_output_coordinate == "native"


def test_uniform_fixture_and_boundary_oracle() -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, validation_count=32,
                         device="cpu", data_law="uniform")
    fixture = make_fixture(config)
    assert fixture["kind"] == "rank_d_uniform_exact_manifold"
    assert fixture["latent_half_width"] == pytest.approx(12.0**0.5)
    assert bool((fixture["query_latents"].abs() <= fixture["latent_half_width"] + 1e-6).all())
    basis = fixture["tangent_basis"].double()
    x = torch.zeros(1, 2, dtype=torch.float64)
    x[0, 0] = fixture["latent_half_width"]
    query = x @ basis.T
    result = population_posterior(query, 0.01, fixture)
    shift = ((query - result) @ basis)[0]
    assert float(shift[0]) == pytest.approx(0.01 * (2 / torch.pi)**0.5, abs=2e-6)
    assert abs(float(shift[1])) < 1e-6


def test_uniform_training_and_native_output_control(tmp_path) -> None:
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=16, depth=1,
                         batch_size=8, steps=2, validation_count=8, log_interval=2,
                         checkpoints=(0, 2), device="cpu", schedules=("log_noise",), data_law="uniform")
    baseline = train_pilot(tmp_path / "data", config)
    intervention = train_pilot(tmp_path / "native", replace(config, output_coordinate="native"))
    assert baseline["final_state_hashes"] == intervention["final_state_hashes"]

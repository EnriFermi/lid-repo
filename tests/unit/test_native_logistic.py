"""Logistic-source flow matching: noise law, heads, two cores and the Full identity."""
import math

import pytest
import torch

from experiments.fair_protocol import geometry, native_contracts, resolve
from models import native_tasks, training
from models.native_logistic import S0, kernel_score, logistic_noise
from models.neural_fields import exact_divergence


def _model(dimension=6, steps=32):
    geo = geometry('e8_gaussian4_pca', 'coefficients', (dimension,))
    config, resolved = resolve('logistic_flowmatching', geo, rank=dimension, device='cpu',
                               steps=steps, preflight=True)
    architecture = training._bottleneck_architecture_from_training_config(
        config, family='logistic_flow', ambient_dim=dimension)
    torch.manual_seed(0)
    return native_tasks.build_field(architecture, 'logistic_flow', config), config, resolved


def test_noise_law_is_unit_variance_logistic():
    generator = torch.Generator().manual_seed(0)
    sample = logistic_noise((200000, 2), device='cpu', dtype=torch.float64, generator=generator)
    assert abs(float(sample.mean())) < 0.02
    assert abs(float(sample.var())-1) < 0.02
    # excess kurtosis of the logistic law is 6/5, far from the Gaussian value 0
    standard = sample/sample.std()
    assert abs(float((standard**4).mean())-3-1.2) < 0.1
    # psi is the score of the sampled density and is bounded by 1/S0
    z = torch.linspace(-4, 4, 33, dtype=torch.float64).requires_grad_(True)
    density = torch.exp(-z/S0)/(S0*(1+torch.exp(-z/S0))**2)
    score = torch.autograd.grad(density.log().sum(), z)[0]
    assert torch.allclose(kernel_score(z.detach()), -score, atol=1e-10)
    assert float(kernel_score(torch.tensor([1e3], dtype=torch.float64))) == pytest.approx(1/S0)


def test_two_cores_share_the_backbone_and_train_both_heads():
    model, config, resolved = _model()
    reference = sum(p.numel() for p in model.core.parameters())
    assert sum(p.numel() for p in model.parameters()) == 2*reference
    assert resolved['actual_parameters'] == 2*resolved['reference_parameters']
    clean = torch.randn(64, model.ambient_dim)
    loss = native_tasks.objective(model, clean, config, torch.Generator().manual_seed(1))
    loss.backward()
    for core in (model.core, model.score_core):
        gradients = [p.grad for p in core.parameters() if p.grad is not None]
        assert gradients and any(float(g.abs().max()) > 0 for g in gradients)


def test_batch_targets_match_the_declared_regression():
    model, config, _ = _model()
    clean = torch.randn(32, model.ambient_dim, dtype=torch.float32)
    point, condition, velocity, score = model.batch(clean, torch.Generator().manual_seed(2))
    time = condition/999.
    assert float(time.min()) >= 0 and float(time.max()) <= 1
    noise = (point-time[:, None]*clean)/(1-time)[:, None]
    assert torch.allclose(velocity, clean-noise, atol=1e-4)
    assert torch.allclose(score, kernel_score(noise), atol=1e-5)


def test_full_readout_uses_the_kernel_score_identity():
    model, config, _ = _model()
    model.eval()
    query = torch.randn(8, model.ambient_dim)
    scale = 0.5
    mean = torch.zeros(model.ambient_dim)
    full = native_tasks.predict(model, query, scale, mean, 1.0, 'full', 'exact', 0, 0, 8)
    response = native_tasks.predict(model, query, scale, mean, 1.0, 'response', 'exact', 0, 0, 8)
    adapter = native_tasks.PosteriorReadout(model)
    with torch.no_grad():
        q = adapter(query, scale)
        correction = (((query-q)/scale)*model.posterior_kernel_score(query, scale)).sum(1)
        gaussian = ((q-query)/scale).square().sum(1)
    assert torch.allclose(torch.as_tensor(full-response, dtype=torch.float64),
                          correction.double(), atol=1e-5)
    # the Gaussian squared-displacement formula is a different quantity here
    assert not torch.allclose(correction, gaussian, atol=1e-3)
    # the posterior is the native flow-matching conversion at lambda = (1-t)/t
    time = 1/(1+scale)
    with torch.no_grad():
        expected = time*query+(1-time)*model(time*query, torch.full((len(query),), 999*time))
    assert torch.allclose(q, expected, atol=0)


def test_scale_window_matches_the_common_grid():
    _, config, _ = _model()
    assert native_tasks.scale_support(config) == (0., math.inf)
    from experiments.fair_measurements import model_scales
    scales = model_scales(config)
    assert len(scales) == 29 and scales[0] == pytest.approx(1/256) and scales[-1] == pytest.approx(64)


def test_contract_records_the_two_head_interface():
    _, config, _ = _model()
    contract = native_tasks.contract(config, 'logistic_flow')
    assert contract['raw_network_output'] == 'velocity_and_kernel_score'
    assert contract['logistic']['cores'] == 2
    assert contract['logistic']['source'] == 'independent_product_logistic_unit_variance'
    assert native_contracts()['logistic_flowmatching']['family'] == 'logistic_flow'
    # other variants keep their stored contracts unchanged
    other = training.TrainingConfig.from_mapping(dict(config.to_dict(), native_variant='rectified_flow'))
    assert 'logistic' not in native_tasks.contract(other, 'rectified_flow')

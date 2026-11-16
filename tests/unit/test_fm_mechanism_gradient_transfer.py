import copy

import torch

from experiments.fm_mechanism_gradient_transfer import adam_metric, block_name, masked_direction, inner_product


def test_conditioning_mask_is_additive_and_does_not_renormalize():
    names = ["input.weight", "condition_embedding.projection.0.weight", "blocks.0.condition.weight", "output.2.bias"]
    direction = [torch.tensor([float(i + 1)], dtype=torch.float64) for i in range(4)]
    without = masked_direction(names, direction, "without_conditioning")
    only = masked_direction(names, direction, "conditioning_only")
    assert [block_name(n) for n in names] == ["input", "conditioning", "conditioning", "output"]
    for a, b, total in zip(without, only, direction):
        torch.testing.assert_close(a + b, total, atol=0, rtol=0)
    assert float(only[1]) == 2 and float(only[2]) == 3


def test_adam_metric_preserves_state_and_uses_bias_correction():
    model = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model(torch.ones(4, 3, dtype=torch.float64)).square().sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    original = copy.deepcopy(state)
    _, params, diagonal = adam_metric(model, state)
    for i, p, metric in zip(state["param_groups"][0]["params"], params, diagonal):
        moments = original["state"][i]
        group = state["param_groups"][0]
        expected = (moments["exp_avg_sq"] / (1 - group["betas"][1] ** float(moments["step"]))).sqrt().add(group["eps"]).reciprocal()
        torch.testing.assert_close(metric, expected, atol=0, rtol=0)
        torch.testing.assert_close(state["state"][i]["exp_avg_sq"], moments["exp_avg_sq"], atol=0, rtol=0)


def test_local_gradient_transfer_sign_and_half_step_on_quadratic():
    theta = torch.tensor([1., 2.], dtype=torch.float64)
    target = torch.tensor([-1., 1.], dtype=torch.float64)
    h = theta - target
    direction = torch.tensor([-2., .1], dtype=torch.float64)
    eta = 1e-5
    before = .5 * (theta - target).square().sum()
    errors = []
    assert inner_product([h], [direction]) < 0
    for fraction in (1., .5):
        changed = .5 * (theta - fraction * eta * direction - target).square().sum() - before
        predicted = -fraction * eta * inner_product([h], [direction])
        assert changed > 0
        errors.append(abs(float(changed) - predicted))
    assert errors[1] < .3 * errors[0]

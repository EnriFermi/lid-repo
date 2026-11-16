import numpy as np
from experiments.response_full_composition_oracle_20260907 import compose, halfline, sphere_field, exp_field


def test_affine_gain_squared_and_bias_accumulated():
    b = np.diag([1., .2, -.1]); c = np.array([.1, .3, -.2])
    field = lambda x, sigma: (b@x+c, b)
    out = compose(field, np.zeros(3), .5)
    np.testing.assert_allclose(out['composed'], (np.eye(3)+b)@c)
    np.testing.assert_allclose(out['composed_jacobian'], b@b)


def test_halfline_full_boundary_exact_but_composition_is_not():
    out = compose(halfline, [0.], 1.)
    np.testing.assert_allclose(out['full'], 1., atol=1e-14)
    assert out['composed_full'] > 1.5
    assert out['composed_response'] < out['response']


def test_composed_chain_rule_is_second_derivative_at_first_mean():
    for field, x, sigma in [(halfline, np.array([.2]), .4),
                            (sphere_field(2), np.array([1., .2, -.1]), .5)]:
        out = compose(field, x, sigma)
        columns = []
        for direction in np.eye(len(x)):
            eps = 1e-5
            plus = compose(field, x+eps*direction, sigma)['composed']
            minus = compose(field, x-eps*direction, sigma)['composed']
            columns.append((plus-minus)/(2*eps))
        np.testing.assert_allclose(np.stack(columns, axis=1), out['composed_jacobian'], atol=1e-8)


def test_sphere2_good_single_full_gets_finite_bias_after_composition():
    out = compose(sphere_field(2), [1., 0., 0.], .1)
    np.testing.assert_allclose(out['full'], 2., atol=1e-10)
    assert out['composed_full'] < 1.995


def test_actual_exp_composition_chain_rule():
    x = np.array([0., .04, .04]); sigma = .044
    field = exp_field()
    out = compose(field, x, sigma)
    columns = []
    for direction in np.eye(3):
        eps = 1e-6
        plus = compose(field, x+eps*direction, sigma)['composed']
        minus = compose(field, x-eps*direction, sigma)['composed']
        columns.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(np.stack(columns, axis=1), out['composed_jacobian'], atol=2e-8)

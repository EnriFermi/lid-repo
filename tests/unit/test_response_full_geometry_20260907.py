import numpy as np

from experiments.response_full_geometry_20260907 import closest_spaghetti, closest_tube
from experiments.response_full_oracle_20260907 import tube_point
from experiments.response_full_oracle_20260907_spaghetti import harmonic_curve


def test_tube_support_and_ambient_distance_are_exact():
    u = np.array([0., 1., 4., 7., 8.])
    theta = np.array([0., .2, 2., 4., 6.])
    points = np.zeros((len(u), 30)); points[:, :3] = tube_point(u, theta)
    points[:, 4] = .125
    result = closest_tube(points)
    np.testing.assert_allclose(result['u'], u, atol=1e-12)
    np.testing.assert_allclose(result['active_distance_squared'], 0., atol=1e-23)
    np.testing.assert_allclose(result['distance_squared'], .125**2, atol=1e-23)


def test_tube_global_candidates_beat_dense_grid_and_keep_axis_ambiguity():
    points = np.array([[-5., 0., 0.], [-3., 0., 3.5], [3., 1., 2.],
                       [-7., 2., 4.], [8., 0., 0.]])
    result = closest_tube(points)
    u = np.linspace(0., 8., 20001)
    dense = (u[None, :]-4-points[:, 0, None])**2+(3*np.exp(-u)[None, :]-np.linalg.norm(points[:, 1:], axis=1)[:, None])**2
    assert np.all(result['distance_squared'] <= dense.min(1)+1e-12)
    assert result['ambiguous_angle'][0]
    assert result['ambiguous_angle'][-1]


def test_spaghetti_continuous_projection_and_ambient_component():
    phase = np.array([.023, .44, 1.7, 3.2, 6.1])
    points = np.zeros((len(phase), 30)); points[:, :20] = harmonic_curve(phase)
    points[:, 20] = -.08
    result = closest_spaghetti(points)
    np.testing.assert_allclose(result['phase'], phase, atol=1e-11)
    np.testing.assert_allclose(result['active_distance_squared'], 0., atol=1e-20)
    np.testing.assert_allclose(result['distance_squared'], .08**2, atol=1e-20)


def test_spaghetti_origin_has_multiple_nearest_phases():
    result = closest_spaghetti(np.zeros((1, 30)))
    assert result['ambiguous_projection'][0]
    assert result['candidate_count'][0] >= 2
    assert result['distance_squared'][0] < 1e-20


def test_spaghetti_off_support_projection_has_stationary_distance():
    rng = np.random.default_rng(3)
    points = harmonic_curve(np.array([.33, .77, 2.1])) + .03*rng.normal(size=(3, 20))
    result = closest_spaghetti(points)
    phase = result['phase']; k = np.arange(2, 22)
    derivative = ((result['nearest_raw']-points)*k*np.cos(phase[:, None]*k)).sum(1)
    np.testing.assert_allclose(derivative, 0., atol=2e-10)
    assert result['grid_refinement_squared_distance_error'].max() < 1e-11


def test_spaghetti_agrees_with_independent_trigonometric_polynomial_roots():
    rng = np.random.default_rng(9)
    points = harmonic_curve(np.array([.17, 1.1, 4.4])) + .2*rng.normal(size=(3, 20))
    result = closest_spaghetti(points)
    for index, query in enumerate(points):
        # g=f'/2 is a real trigonometric polynomial of degree42. Multiply
        # its Laurent polynomial byz^42 and enumerate all complex roots.
        coefficients = np.zeros(85, dtype=np.complex128)
        for q, k in zip(query, range(2, 22)):
            coefficients[42+2*k] += k/(4j)
            coefficients[42-2*k] -= k/(4j)
            coefficients[42+k] -= q*k/2
            coefficients[42-k] -= q*k/2
        roots = np.polynomial.polynomial.polyroots(coefficients)
        phases = np.angle(roots[np.abs(np.abs(roots)-1) < 1e-5])
        polynomial_minimum = np.square(harmonic_curve(phases)-query).sum(1).min()
        np.testing.assert_allclose(result['distance_squared'][index], polynomial_minimum, atol=1e-10, rtol=1e-10)

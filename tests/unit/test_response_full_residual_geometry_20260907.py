import numpy as np

from experiments.response_full_residual_geometry_20260907 import symmetric_mass


def test_zero_trace_does_not_certify_small_matrix_error():
    matrix=np.diag([.5,-.5,0.])
    parts=symmetric_mass(matrix)
    assert parts["positive"]==parts["negative"]==.5
    assert np.trace(matrix)==0
    assert (matrix**2).sum()==.5


def test_symmetrized_mass_ignores_skew_but_not_its_matrix_energy():
    matrix=np.array([[.2,.8],[-.8,-.3]])
    parts=symmetric_mass(matrix)
    np.testing.assert_allclose(parts["positive"],.2)
    np.testing.assert_allclose(parts["negative"],.3)
    np.testing.assert_allclose(parts["positive"]-parts["negative"],np.trace(matrix))
    assert (matrix**2).sum()>.2**2+.3**2

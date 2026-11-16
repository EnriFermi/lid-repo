import numpy as np
from experiments.response_full_oracle_20260907_eigen import gated_correction
from experiments.response_full_oracle_20260907_half import halfline_readouts


def test_gate_preserves_every_clean_flat_halfspace_correction():
    for a in np.linspace(0,8,101):
        r=halfline_readouts(a)
        c,_,_,_=gated_correction(np.array([np.sqrt(r['correction'])]),1.,np.array([[r['response']]]))
        np.testing.assert_allclose(c,r['correction'],atol=1e-14,rtol=1e-14)


def test_gate_is_rotation_invariant_and_excludes_unresponsive_displacement():
    shift=np.array([2.,3.]);cov=np.diag([.01,.8]);angle=.73
    orthogonal=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    a=gated_correction(shift,1.,cov)[0]
    b=gated_correction(orthogonal@shift,1.,orthogonal@cov@orthogonal.T)[0]
    np.testing.assert_allclose(a,9.,atol=1e-14)
    np.testing.assert_allclose(a,b,atol=1e-13)

"""Analytic checks for the fixed-checkpoint parity diagnostic."""

import torch

from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_parity_probe import energy, sideways


def test_sideways_antipodal_energy_splits_exactly():
    generator = torch.Generator().manual_seed(314)
    x = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    odd = x @ torch.tensor([[1., .2, 0.], [0., .9, .1], [.3, 0., 1.]], dtype=x.dtype)
    even = .1 + .03*x.square()
    u = x / x.norm(dim=-1, keepdim=True)
    left = (energy(sideways(odd+even,u),.316)+energy(sideways(-odd+even,u),.316))/2
    right = energy(sideways(odd,u),.316)+energy(sideways(even,u),.316)
    torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)


def test_odd_symmetrization_has_averaged_jacobian_and_removes_even_correction():
    x = torch.tensor([[.2,-.4],[1.1,.3]],dtype=torch.float64)

    def fn(z):
        return 1.2*z + .3*z.square() + .1

    vp,jp = value_and_jacobian(fn,x)
    vm,jm = value_and_jacobian(fn,-x)
    odd,jo = value_and_jacobian(lambda z:(fn(z)-fn(-z))/2,x)
    even=(vp+vm)/2
    torch.testing.assert_close(jo,(jp+jm)/2)
    torch.testing.assert_close(odd,1.2*x)
    torch.testing.assert_close((energy(vp-x,.2)+energy(vm+x,.2))/2,
                               energy(odd-x,.2)+energy(even,.2))

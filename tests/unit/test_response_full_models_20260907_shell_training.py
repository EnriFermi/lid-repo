import torch

from experiments.response_full_models_20260907_shell_training import mix_normal_noise
from experiments.response_full_training_20260907 import objective
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


def test_noising_preserves_shared_global_span_and_unmodified_half_exactly():
    torch.manual_seed(73)
    basis=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q
    z=torch.randn(12,7,dtype=torch.float64)
    mixed=mix_normal_noise(z,basis,.25)
    torch.testing.assert_close(mixed@basis,z@basis,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(mixed[1::2],z[1::2],rtol=0,atol=0)
    normal=z-(z@basis)@basis.T
    torch.testing.assert_close(mixed[::2]-(mixed[::2]@basis)@basis.T,.25*normal[::2],atol=1e-14,rtol=1e-14)
    assert mix_normal_noise(z,basis,1.) is z


def test_mixed_noise_ordinary_clean_target_loss_and_parameter_gradients():
    torch.manual_seed(74)
    model=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4,zero_init_output=False)).double()
    x=torch.randn(6,4,dtype=torch.float64);x[:,2:]=0
    z=mix_normal_noise(torch.randn_like(x),torch.eye(4,dtype=torch.float64)[:,:2])
    lam=torch.linspace(.03,.7,6,dtype=torch.float64)
    loss=objective(model,x,z,lam,"noise")
    raw=model((x+lam[:,None]*z)/torch.sqrt(1+lam[:,None]**2),lam.log())
    expected=(raw-z).square().mean()
    torch.testing.assert_close(loss,expected)
    first=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    second=torch.autograd.grad(expected,tuple(model.parameters()))
    for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)


def test_projection_posterior_zero_risk_in_globally_absent_directions_all_amplitudes():
    torch.manual_seed(75)
    basis=torch.eye(6,dtype=torch.float64)[:,:2]
    x=torch.randn(8,2,dtype=torch.float64)@basis.T
    z=torch.randn_like(x)
    for amplitude in [.1,.25,1.]:
        observed=x+.3*mix_normal_noise(z,basis,amplitude)
        oracle=(observed@basis)@basis.T
        torch.testing.assert_close(oracle[:,2:],torch.zeros(8,4,dtype=torch.float64),atol=0,rtol=0)

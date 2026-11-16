import math

import torch

from experiments.fm_mechanism_condition_refit import condition_override, fit_vectors, training_target
from experiments.fm_mechanism_pilot import canonical_posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def test_condition_override_identity_cleanup_and_only_vector_gradients():
    torch.manual_seed(51)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=16,
        depth=1,condition_dim=64,zero_init_output=False)).double()
    for p in field.parameters():p.requires_grad_(False)
    x=torch.randn(7,4,dtype=torch.float64);lam=.316227766
    e=field.condition_embedding(x.new_tensor(math.log(lam)))[0].detach()
    original=canonical_posterior(field,x,lam,"rectified_linear")
    vector=torch.nn.Parameter(e.clone())
    with condition_override(field,vector):
        changed=canonical_posterior(field,x,lam,"rectified_linear")
        torch.testing.assert_close(changed,original,atol=1e-12,rtol=1e-12)
        changed.square().mean().backward()
    assert vector.grad is not None and vector.grad.norm()>0
    assert all(p.grad is None for p in field.parameters())
    assert not field.condition_embedding._forward_hooks
    torch.testing.assert_close(canonical_posterior(field,x,lam,"rectified_linear"),original,atol=0,rtol=0)


def test_condition_vector_fit_keeps_checkpoint_parameters_fixed():
    torch.manual_seed(53)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=16,
        depth=1,condition_dim=64,zero_init_output=False)).double()
    for p in field.parameters():p.requires_grad_(False)
    original={k:v.clone() for k,v in field.state_dict().items()}
    x=torch.randn(16,4,dtype=torch.float64);lam=1.
    e=field.condition_embedding(x.new_tensor(math.log(lam)))[0].detach()
    with condition_override(field,e+.1):
        target=canonical_posterior(field,x,lam,"log_noise").detach()
    vector,receipt=fit_vectors(field,"log_noise",{lam:(x,target)},{lam:e},shared=False,max_iter=10)
    assert receipt["final_loss"]<receipt["initial_loss"]
    assert vector.shape==(64,)
    assert all(torch.equal(v,original[k]) for k,v in field.state_dict().items())
    assert not field.condition_embedding._forward_hooks


def test_shared_condition_delta_uses_original_scale_specific_embeddings():
    torch.manual_seed(57)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=16,
        depth=1,condition_dim=64,zero_init_output=False)).double()
    for p in field.parameters():p.requires_grad_(False)
    x=torch.randn(16,4,dtype=torch.float64)
    bases={lam:field.condition_embedding(x.new_tensor(math.log(lam)))[0].detach()
           for lam in (.316227766,1.)}
    batches={}
    for lam,base in bases.items():
        with condition_override(field,base+.05):
            target=canonical_posterior(field,x,lam,"rectified_linear").detach()
        batches[lam]=(x,target)
    vector,receipt=fit_vectors(field,"rectified_linear",batches,bases,shared=True,max_iter=10)
    assert receipt["final_loss"]<receipt["initial_loss"]
    assert vector.shape==(64,) and vector.norm()>0
    assert not field.condition_embedding._forward_hooks


def test_finite_clean_target_never_calls_population_oracle():
    clean=torch.randn(5,4,dtype=torch.float64)
    inputs=clean+torch.randn_like(clean)
    def forbidden_oracle(*args):
        raise AssertionError("clean_X fitting must not use oracle supervision")
    assert training_target(clean,inputs,1.,forbidden_oracle,"clean_X") is clean

import copy
import torch
from experiments.response_full_models_20260907_ve_refinement import edm_readout,original_loss,parser
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField
from models.training import diffusion_denoiser_output


def test_fast_edm_log_condition_value_trace_matches_exact_autograd():
    torch.manual_seed(80)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=8,depth=2,
        condition_dim=8,fourier_features=4,condition_transform="log",zero_init_output=False)).double().eval()
    x=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    for sigma in [.01,.316,3.]:
        b,r,c=edm_readout(field,x,sigma)
        truth=diffusion_denoiser_output(field,x,sigma,preconditioning="edm")
        trace=sum(torch.autograd.grad(truth[:,i].sum(),x,retain_graph=True)[0][:,i] for i in range(4))
        torch.testing.assert_close(b,truth,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(r,trace,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(c,(truth-x).square().sum(-1)/sigma**2)


def test_shared_original_loss_generators_give_identical_values_and_gradients():
    torch.manual_seed(81)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4,condition_transform="log",zero_init_output=False)).double()
    other=copy.deepcopy(field);x=torch.randn(7,4,dtype=torch.float64)
    cfg={"sigma_min":.01,"sigma_max":16.,"diffusion_loss_weighting":"edm_x0",
        "diffusion_data_std":1.,"diffusion_preconditioning":"edm"}
    generators=[torch.Generator().manual_seed(82) for _ in range(2)]
    losses=[original_loss(model,x,cfg,g) for model,g in zip([field,other],generators)]
    torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    for loss in losses:loss.backward()
    for a,b in zip(field.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert torch.equal(generators[0].get_state(),generators[1].get_state())


def test_average_function_full_includes_average_response_not_average_correction():
    torch.manual_seed(83)
    b=torch.randn(16,4,7,5,dtype=torch.float64);x=torch.randn(7,5,dtype=torch.float64)
    r=torch.randn(16,4,7,dtype=torch.float64);s=torch.tensor([.01,.1,.316,1.])[:,None]
    mean_b=b.mean(0);c=(mean_b-x).square().sum(-1)/s**2
    snapshot_c=(b-x).square().sum(-1)/s**2
    variance=(b-mean_b).square().sum(-1).mean(0)/s**2
    torch.testing.assert_close(snapshot_c.mean(0),c+variance)
    torch.testing.assert_close((r+snapshot_c).mean(0),r.mean(0)+c+variance)


def test_dataset_extension_preserves_default_protocol_and_correct_tube_dimension():
    default=parser().parse_args(["--output","unused"])
    tube=parser().parse_args(["--output","unused","--dataset","e6_exp_pca"])
    assert default.dataset=="e8_spaghetti_pca"
    assert DATASET_SPECS[default.dataset].true_lid==1
    assert DATASET_SPECS[tube.dataset].true_lid==2
    for field in ["steps","average_steps","low_lr","seed","device"]:
        assert getattr(default,field)==getattr(tube,field)

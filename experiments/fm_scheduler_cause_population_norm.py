"""Conditional population denominator: a batch-separable stateful normalizer.

The forward uses only existing buffers. Current-batch hidden variances are
detached observations committed AFTER optimizer.step, never a stopped-gradient
version of the current sample's LayerNorm denominator.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from experiments.fm_mechanism_diagnostics import alpha_at
from experiments.fm_mechanism_fast_trace import (_layer_norm_value_jacobian,
    _linear_jacobian, _silu_value_jacobian)


class PopulationRMSNorm(nn.Module):
    def __init__(self, original, *, log_grid, decay=.9):
        super().__init__()
        if not isinstance(original,nn.LayerNorm) or len(original.normalized_shape)!=1:
            raise TypeError("Expected one-dimensional affine LayerNorm")
        if original.weight is None or original.bias is None or not 0<=decay<1:
            raise ValueError("Affine parameters and valid EMA decay are required")
        self.normalized_shape=original.normalized_shape
        self.eps=original.eps
        self.decay=float(decay)
        self.weight=nn.Parameter(original.weight.detach().clone())
        self.bias=nn.Parameter(original.bias.detach().clone())
        grid=torch.as_tensor(log_grid,dtype=original.weight.dtype,device=original.weight.device)
        if grid.ndim!=1 or len(grid)<2 or not bool((grid[1:]>grid[:-1]).all()):
            raise ValueError("Log grid must be strictly increasing")
        self.register_buffer("log_grid",grid)
        self.register_buffer("running_variance",torch.ones_like(grid))
        self.register_buffer("observation_count",torch.zeros_like(grid))
        self.register_buffer("updates",torch.zeros((),dtype=torch.int64,device=grid.device))
        self.capture_statistics=False
        self._log_scales=None
        self._pending=None

    def brackets(self,log_scales):
        grid=self.log_grid.to(log_scales)
        values=log_scales.reshape(-1).clamp(grid[0],grid[-1])
        right=torch.searchsorted(grid,values,right=True).clamp(1,len(grid)-1)
        left=right-1
        weight=(values-grid[left])/(grid[right]-grid[left])
        return left,right,weight

    def variance_at(self,log_scales):
        left,right,weight=self.brackets(log_scales)
        variance=self.running_variance.to(log_scales)
        return (1-weight)*variance[left]+weight*variance[right]

    def forward_at(self,value,log_scales,*,capture=False):
        logs=torch.as_tensor(log_scales,dtype=value.dtype,device=value.device)
        if logs.ndim==0:logs=logs.expand(len(value))
        logs=logs.reshape(-1)
        if len(logs)!=len(value):raise ValueError("Scale batch mismatch")
        centered=value-value.mean(-1,keepdim=True)
        denominator=(self.variance_at(logs)+self.eps).sqrt()
        if capture:
            self._pending=(logs.detach(),centered.detach().square().mean(-1))
        return centered/denominator[:,None]*self.weight+self.bias

    def forward(self,value):
        if self._log_scales is None:raise RuntimeError("Population normalizer needs log-lambda context")
        return self.forward_at(value,self._log_scales,capture=self.training or self.capture_statistics)

    def take_observations(self):
        if self._pending is None:raise RuntimeError("No pending normalization observations")
        result=self._pending
        self._pending=None
        return result

    def weighted_statistics(self,log_scales,variance):
        left,right,weight=self.brackets(log_scales)
        size=len(self.log_grid)
        counts=torch.zeros(size,dtype=torch.float64,device=variance.device)
        sums=torch.zeros_like(counts)
        squares=torch.zeros_like(counts)
        for index,w in ((left,1-weight),(right,weight)):
            counts.scatter_add_(0,index,w.double())
            sums.scatter_add_(0,index,w.double()*variance.double())
            squares.scatter_add_(0,index,w.double()*variance.double().square())
        return counts,sums,squares

    @torch.no_grad()
    def commit_statistics(self):
        """Caller must invoke only after backward AND optimizer.step."""
        logs,variance=self.take_observations()
        counts,sums,_=self.weighted_statistics(logs,variance)
        present=counts>0
        estimates=(sums/counts.clamp_min(1)).to(self.running_variance)
        old=self.running_variance[present]
        self.running_variance[present]=self.decay*old+(1-self.decay)*estimates[present]
        self.observation_count.add_(counts.to(self.observation_count))
        self.updates.add_(1)


def population_grid(lambda_min,lambda_max,bins=16,anchors=(.1,.316227766,1.)):
    if not 0<lambda_min<lambda_max or bins<2:raise ValueError("Invalid population grid")
    base=torch.linspace(math.log(lambda_min),math.log(lambda_max),bins,dtype=torch.float64)
    extra=torch.tensor([math.log(x) for x in anchors if lambda_min<=x<=lambda_max],dtype=torch.float64)
    return torch.unique(torch.cat((base,extra)),sorted=True)


def _set_population_condition(field,args,kwargs):
    condition=args[1] if len(args)>1 else kwargs.get("condition")
    if condition is None:raise ValueError("Missing scalar conditioning")
    field.output[0]._log_scales=condition


def install_population_norm(field,config):
    if isinstance(field.output[0],PopulationRMSNorm):raise ValueError("Population norm already installed")
    field.output[0]=PopulationRMSNorm(field.output[0],log_grid=config["log_grid"],decay=config["decay"])
    field.register_forward_pre_hook(_set_population_condition,with_kwargs=True)
    return field


@torch.no_grad()
def calibrate_population(field,arm,config,fixture,*,per_knot,seed,device,anchor,predict,draw_batch):
    """Fresh paired population calibration, with zero mutation of training RNG."""
    norm=field.output[0]
    if not isinstance(norm,PopulationRMSNorm) or per_knot<=0:raise ValueError("Invalid calibration")
    generator=torch.Generator(device=device).manual_seed(seed)
    was_training=field.training
    field.eval()
    norm.capture_statistics=True
    total_count=torch.zeros_like(norm.log_grid,dtype=torch.float64)
    total_sum=torch.zeros_like(total_count)
    total_square=torch.zeros_like(total_count)
    try:
        for log_scale in norm.log_grid:
            remaining=per_knot
            while remaining:
                count=min(config["batch_size"],remaining)
                draw_config={**config,"batch_size":count}
                clean,noise,_=draw_batch(draw_config,fixture,generator,device)
                scale=log_scale.exp().expand(count)
                predict(field,clean+scale[:,None]*noise,scale,arm,anchor)
                logs,variance=norm.take_observations()
                counts,sums,squares=norm.weighted_statistics(logs,variance)
                total_count.add_(counts)
                total_sum.add_(sums)
                total_square.add_(squares)
                remaining-=count
        means=total_sum/total_count
        sample_variance=(total_square/total_count-means.square()).clamp_min(0)
        standard_error=(sample_variance/total_count).sqrt()
        norm.running_variance.copy_(means.to(norm.running_variance))
        norm.observation_count.copy_(total_count.to(norm.observation_count))
        norm.updates.zero_()
        return {"seed":seed,"examples_per_knot":per_knot,"total_examples":per_knot*len(norm.log_grid),
            "log_grid":norm.log_grid.cpu().tolist(),"variance":means.cpu().tolist(),
            "variance_standard_error":standard_error.cpu().tolist(),
            "relative_rms_standard_error":(.5*standard_error/means).cpu().tolist(),
            "note":"Forward-only separate private stream; pointwise conditioning-grid means; no evaluation/query adaptation"}
    finally:
        norm.capture_statistics=False
        norm._pending=None
        field.train(was_training)


def population_value_and_trace(field,canonical_inputs,noise_ratio,schedule):
    """Exact canonical value/trace, including the fixed conditional denominator."""
    norm=field.output[0]
    if not isinstance(norm,PopulationRMSNorm):raise TypeError("Expected population final normalization")
    if canonical_inputs.dtype!=torch.float64 or field.input.weight.dtype!=torch.float64:
        raise ValueError("Exact population trace requires float64")
    batch=len(canonical_inputs)
    flattened=canonical_inputs.reshape(batch,-1)
    lam=torch.as_tensor(noise_ratio,dtype=flattened.dtype,device=flattened.device)
    if lam.ndim==0:lam=lam.expand(batch)
    if lam.shape!=(batch,) or not bool((lam>0).all()):raise ValueError("Invalid scale")
    alpha=alpha_at(lam,schedule)
    embedding=field.condition_embedding(lam.log())
    state=field.input(alpha[:,None]*flattened)+field.input_condition(embedding)
    jacobian=field.input.weight[None].expand(batch,-1,-1)*alpha[:,None,None]
    for block in field.blocks:
        normalized,nj=_layer_norm_value_jacobian(block.norm,state,jacobian)
        preactivation=block.hidden(normalized)+block.condition(embedding)
        activated,uj=_silu_value_jacobian(preactivation,_linear_jacobian(block.hidden,nj))
        state=(state+block.output(activated))*2**-.5
        jacobian=(jacobian+_linear_jacobian(block.output,uj))*2**-.5
    centered=state-state.mean(-1,keepdim=True)
    centered_jacobian=jacobian-jacobian.mean(1,keepdim=True)
    inverse_std=(norm.variance_at(lam.log())+norm.eps).rsqrt()
    state=centered*inverse_std[:,None]*norm.weight+norm.bias
    jacobian=centered_jacobian*inverse_std[:,None,None]*norm.weight[None,:,None]
    state,jacobian=_silu_value_jacobian(state,jacobian)
    head=field.output[-1]
    return head(state).reshape_as(canonical_inputs),torch.einsum("ih,bhi->b",head.weight,jacobian)

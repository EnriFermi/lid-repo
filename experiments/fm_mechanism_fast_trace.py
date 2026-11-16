"""Exact differentiable posterior value/trace for the pilot's residual MLP.

Propagate B x hidden x input Jacobians and contract the final output head;
never materialize the B x ambient x ambient output Jacobian. Scalar condition
is held fixed with respect to spatial differentiation. Caller owns dtype,
device, evaluation mode and any parameter gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments.fm_mechanism_diagnostics import alpha_at


def _linear_jacobian(layer: nn.Linear, jacobian: Tensor) -> Tensor:
    return torch.einsum("oh,bhi->boi", layer.weight, jacobian)


def _layer_norm_value_jacobian(layer: nn.Module, value: Tensor, jacobian: Tensor) -> tuple[Tensor, Tensor]:
    if isinstance(layer, nn.Identity):
        return value, jacobian
    if not isinstance(layer, nn.LayerNorm) or tuple(layer.normalized_shape) != (value.shape[-1],):
        raise TypeError("Only final-channel LayerNorm or Identity is supported")
    centered = value-value.mean(dim=-1, keepdim=True)
    variance_eps = centered.square().mean(dim=-1, keepdim=True)+layer.eps
    inverse_std = variance_eps.rsqrt()
    centered_jacobian = jacobian-jacobian.mean(dim=1, keepdim=True)
    covariance = (centered[:, :, None]*jacobian).mean(dim=1, keepdim=True)
    transformed = (centered_jacobian-centered[:, :, None]*covariance/variance_eps[:, :, None])*inverse_std[:, :, None]
    if layer.weight is not None:
        transformed = transformed*layer.weight[None, :, None]
    return layer(value), transformed


def _silu_value_jacobian(value: Tensor, jacobian: Tensor) -> tuple[Tensor, Tensor]:
    sigmoid = value.sigmoid()
    derivative = sigmoid*(1+value*(1-sigmoid))
    return torch.nn.functional.silu(value), derivative[:, :, None]*jacobian


def fast_value_and_trace(
    field: nn.Module,
    canonical_inputs: Tensor,
    noise_ratio: Tensor | float,
    schedule: str,
    *,
    input_coordinate: str = "native",
    output_coordinate: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return canonical posterior value and exact spatial trace, float64.

    This mirrors ``canonical_posterior`` for ScaleConditionedNeuralField and
    leaves the computational graph intact, including for gradient-of-trace
    calculations. Lambda may be a scalar or one value per query. Active
    dropout is unsupported because its random mask must not be resampled
    independently of the propagated Jacobian.
    """
    if canonical_inputs.ndim < 2 or canonical_inputs.dtype != torch.float64:
        raise ValueError("Inputs must have a batch dimension and float64 dtype")
    if field.input.weight.dtype != torch.float64:
        raise ValueError("Convert the field to float64 before evaluation")
    batch = len(canonical_inputs)
    flattened = canonical_inputs.reshape(batch, -1)
    if flattened.shape[-1] != field.config.ambient_dim:
        raise ValueError("Input ambient dimension does not match the field")
    lam = torch.as_tensor(noise_ratio, dtype=flattened.dtype, device=flattened.device)
    if lam.ndim == 0:
        lam = lam.expand(batch)
    if lam.shape != (batch,) or not bool((torch.isfinite(lam)&(lam > 0)).all()):
        raise ValueError("noise_ratio must be positive scalar or a batch vector")
    if input_coordinate == "native":
        alpha = alpha_at(lam, schedule)
    elif input_coordinate == "channel":
        alpha = torch.ones_like(lam)
    else:
        raise ValueError("input_coordinate must be native or channel")
    embedded = field.condition_embedding(lam.log())
    state = field.input(alpha[:, None]*flattened)+field.input_condition(embedded)
    jacobian = field.input.weight[None].expand(batch, -1, -1)*alpha[:, None, None]
    for block in field.blocks:
        if block.training and getattr(block.dropout, "p", 0) > 0:
            raise ValueError("Active dropout is unsupported; use eval()")
        normalized, normalized_jacobian = _layer_norm_value_jacobian(block.norm, state, jacobian)
        preactivation = block.hidden(normalized)+block.condition(embedded)
        update_jacobian = _linear_jacobian(block.hidden, normalized_jacobian)
        activated, update_jacobian = _silu_value_jacobian(preactivation, update_jacobian)
        update = block.output(activated)
        update_jacobian = _linear_jacobian(block.output, update_jacobian)
        state = (state+update)*(2.0**-.5)
        jacobian = (jacobian+update_jacobian)*(2.0**-.5)
    state, jacobian = _layer_norm_value_jacobian(field.output[0], state, jacobian)
    state, jacobian = _silu_value_jacobian(state, jacobian)
    head = field.output[-1]
    value = head(state)
    trace = torch.einsum("ih,bhi->b", head.weight, jacobian)
    output_coordinate = output_coordinate or getattr(field, "pilot_output_coordinate", "data")
    if output_coordinate == "native":
        output_alpha = alpha_at(lam, schedule)
        value = value/output_alpha[:, None]
        trace = trace/output_alpha
    elif output_coordinate != "data":
        raise ValueError("output_coordinate must be native or data")
    return value.reshape_as(canonical_inputs), trace

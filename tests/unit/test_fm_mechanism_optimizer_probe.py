"""Guard the independent-Adam-state contract of the small late-LR probe."""

import ast
import copy
import inspect

import torch

from experiments import fm_mechanism_optimizer_probe as probe


def test_late_lr_probe_restores_independent_optimizer_state() -> None:
    # The probe is a standalone executable, not a reusable optimizer factory.
    # Guard its restoration call explicitly, then test the guarded operation.
    tree = ast.parse(inspect.getsource(probe.main))
    restore_calls = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "load_state_dict"]
    assert len(restore_calls) == 1
    argument = restore_calls[0].args[0]
    assert isinstance(argument, ast.Call)
    assert isinstance(argument.func, ast.Attribute)
    assert isinstance(argument.func.value, ast.Name)
    assert (argument.func.value.id, argument.func.attr) == ("copy", "deepcopy")

    parameter = torch.nn.Parameter(torch.ones(3))
    original = torch.optim.AdamW([parameter], lr=.001)
    parameter.grad = torch.ones_like(parameter)
    original.step()
    source = copy.deepcopy(original.state_dict())
    restored = []
    for learning_rate in (.001, .0001):
        value = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([value], lr=learning_rate)
        optimizer.load_state_dict(copy.deepcopy(source))
        optimizer.param_groups[0]["lr"] = learning_rate
        restored.append((value, optimizer))
    states = [optimizer.state[value] for value, optimizer in restored]
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert states[0][name].data_ptr() != states[1][name].data_ptr()
        assert states[0][name].data_ptr() != source["state"][0][name].data_ptr()
        torch.testing.assert_close(states[0][name], states[1][name], atol=0, rtol=0)
    value, optimizer = restored[0]
    value.grad = torch.ones_like(value)
    optimizer.step()
    assert states[0]["step"].item() == 2
    assert states[1]["step"].item() == source["state"][0]["step"].item() == 1

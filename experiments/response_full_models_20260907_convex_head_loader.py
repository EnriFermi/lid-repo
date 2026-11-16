"""Precision-preserving loader for the float64 convex-head diagnostic states."""
from pathlib import Path
import torch
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


def load_convex_head(path: str|Path,state: str,device: str="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["schema"]!="ve_convex_head_refit_v1":raise ValueError("Not a convex-head state")
    model=ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    # Loading before this conversion would round the fitted float64 head to
    # float32 silently. Do not use the historical refinement loader here.
    model=model.to(device=device,dtype=torch.float64)
    model.load_state_dict(payload["models"][state],strict=True)
    return model.eval()

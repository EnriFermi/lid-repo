"""One fixed noisy-MSE calibration of the second denoising displacement.

g = b + eta * (b(b)-b), with one unconstrained spatially constant eta per
scale, fitted only to the original1000 validation X times eight corruptions.
Separate supervised validation-scale selection follows; no eta/P/gamma sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_composition_20260907 import ROOT, cases
from experiments.response_full_composition_selection_20260907 import select_scale
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits


def fit_residual_coefficient(first, second, clean):
    delta = second-first
    energy = delta.square().sum()
    if not torch.isfinite(energy):
        raise ValueError("Nonfinite second-step energy")
    if energy == 0:
        return torch.zeros((), dtype=first.dtype, device=first.device)
    coefficient = (delta*(clean-first)).sum()/energy
    if not torch.isfinite(coefficient):
        raise ValueError("Nonfinite fitted coefficient")
    return coefficient


def blend(first, second, coefficient):
    return first+coefficient*(second-first)


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path("experiments/response_full_composition_20260907.py"),
        Path("experiments/response_full_composition_selection_20260907.py"),
        Path("experiments/response_full_models_20260907_vp.py"),
        Path("experiments/response_full_models_20260907_spaghetti_local.py"),
        Path("experiments/response_full_models_20260907_span.py"),
        Path("experiments/response_full_training_20260907.py"),
        Path("experiments/ve_known_lid_recoverability_campaign.py"),
        Path("models/neural_fields.py"),
        ROOT/"two_pass_validation_selection/manifest.json", ROOT/"two_pass_composition/manifest.json"]
    hashes = {str(p): sha256(p) for p in paths}
    rows, choices, gates = [], [], []
    for case in [c for c in cases() if c["family"] == "vp"]:
        fixture_path = case["checkpoint"].parent/"fixture.pt"
        val_path = ROOT/"two_pass_validation_selection"/(case["name"]+".npz")
        test_path = ROOT/"two_pass_composition"/(case["name"]+".npz")
        for path in [case["checkpoint"], fixture_path, val_path, test_path]:
            hashes[str(path)] = sha256(path)
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
        validation, test = np.load(val_path), np.load(test_path)
        cfg = payload["config"]
        raw_train, raw_val, raw_test = make_splits(case["dataset"],
            train_size=cfg["train_count"], validation_size=cfg["selection_count"],
            test_size=cfg["test_count"], data_seed=0)
        raw_hashes = {k: tensor_sha(torch.from_numpy(v)) for k,v in
                     [("train",raw_train),("selection",raw_val),("test",raw_test)]}
        assert raw_hashes == fixture["split_hashes"]
        mean, rms = fixture["mean"], float(fixture["rms"])
        xval = (torch.from_numpy(raw_val)-mean)/rms
        xtest = (torch.from_numpy(raw_test)-mean)/rms
        np.testing.assert_array_equal(xval.numpy(), validation["validation_queries"])
        np.testing.assert_array_equal(xtest[test["query_ids"]].numpy(), test["queries"])
        np.testing.assert_array_equal(validation["scales"], test["scales"])
        model = load_field(payload, case["kind"], "average", args.device)
        qval = xval.double().to(args.device)
        xv = qval.repeat_interleave(8, dim=0)
        xt = xtest.double().to(args.device)
        generator = torch.Generator(device=args.device).manual_seed(20261412)
        output = {}
        def save(key, value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            output.setdefault(key, []).append(value)
        def twice(function, queries):
            first, second = [], []
            with torch.no_grad():
                for batch in queries.split(256):
                    b1 = function(batch)
                    first.append(b1)
                    second.append(function(b1))
            return torch.cat(first), torch.cat(second)
        truth = DATASET_SPECS[case["dataset"]].true_lid
        for i, scale in enumerate(test["scales"]):
            scale = float(scale)
            function = lambda q: posterior(model, q, scale, case["kind"])
            yv = xv+scale*torch.randn(xv.shape, dtype=xv.dtype, device=args.device, generator=generator)
            yt = xt+scale*torch.randn(xt.shape, dtype=xt.dtype, device=args.device, generator=generator)
            bv1, bv2 = twice(function, yv)
            bt1, bt2 = twice(function, yt)
            eta = fit_residual_coefficient(bv1, bv2, xv)
            gv, gt = blend(bv1, bv2, eta), blend(bt1, bt2, eta)
            q1 = torch.as_tensor(validation["base_posterior"][i], device=args.device)
            q2 = torch.as_tensor(validation["posterior"][i], device=args.device)
            rq1 = torch.as_tensor(validation["base_response"][i], device=args.device)
            rq2 = torch.as_tensor(validation["response"][i], device=args.device)
            qg, rg = blend(q1, q2, eta), blend(rq1, rq2, eta)
            cg = (qg-qval).square().sum(-1)/scale**2
            def composed(q):
                first = function(q)
                return blend(first, function(first), eta)
            direct_value, direct_j = active_jacobian(composed, qval[:8], qval.shape[-1])
            direct_r = direct_j.diagonal(dim1=-2, dim2=-1).sum(-1)
            torch.testing.assert_close(direct_value, qg[:8], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(direct_r, rg[:8], rtol=1e-10, atol=1e-10)
            gates.append(dict(case=case["name"], scale=scale,
                direct_value_error=float((direct_value-qg[:8]).abs().max()),
                direct_response_error=float((direct_r-rg[:8]).abs().max())))
            coefficient = float(eta)
            tq = blend(test["base_posterior"][i], test["posterior"][i], coefficient)
            tr = blend(test["base_response"][i], test["response"][i], coefficient)
            tc = ((tq-test["queries"])**2).sum(-1)/scale**2
            risk = lambda values, targets: float((values-targets).square().mean()/scale**2)
            row = dict(case=case["name"], scale=scale, eta=coefficient,
                validation_noisy_risk_base=risk(bv1,xv),
                validation_noisy_risk_two=risk(bv2,xv),
                validation_noisy_risk_blend=risk(gv,xv),
                test_noisy_risk_base=risk(bt1,xt),
                test_noisy_risk_two=risk(bt2,xt),
                test_noisy_risk_blend=risk(gt,xt),
                test_response_mae=float(np.abs(tr-truth).mean()),
                test_full_mae=float(np.abs(tr+tc-truth).mean()))
            assert row["validation_noisy_risk_blend"] <= min(
                row["validation_noisy_risk_base"],row["validation_noisy_risk_two"])+1e-12
            rows.append(row)
            for key, value in [("eta",eta), ("noisy_validation_y",yv),
                ("noisy_validation_first",bv1),("noisy_validation_second",bv2),
                ("noisy_test_y",yt),("noisy_test_first",bt1),("noisy_test_second",bt2),
                ("validation_posterior",qg),("validation_response",rg),
                ("validation_correction",cg),("validation_full",rg+cg),
                ("test_posterior",tq),("test_response",tr),
                ("test_correction",tc),("test_full",tr+tc)]:
                save(key,value)
            print(json.dumps(row), flush=True)
        arrays = {k:np.stack(v) for k,v in output.items()}
        for readout in ["response","full"]:
            selected, risks = select_scale(arrays["validation_"+readout],truth,test["scales"])
            choices.append(dict(case=case["name"], readout=readout,
                selected_index=selected, selected_scale=float(test["scales"][selected]),
                eta=float(arrays["eta"][selected]), validation_mae=float(risks[selected]),
                test_mae=float(np.abs(arrays["test_"+readout][selected]-truth).mean()),
                at_grid_boundary=bool(selected in [0,len(risks)-1]),
                all_scale_validation_mae=risks.tolist()))
        np.savez_compressed(args.output/(case["name"]+".npz"),
            validation_clean=xval.numpy(), noisy_test_clean=xtest.numpy(),
            validation_ids=np.arange(1000), test_query_ids=test["query_ids"],
            test_queries=test["queries"], scales=test["scales"], **arrays)
    assert all(sha256(p)==digest for p,digest in hashes.items())
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output/"selection.json").write_text(json.dumps(choices,indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(dict(
        completed=True,sources=hashes,sources_unchanged=True,direct_checks=gates,
        coefficient="Unconstrained least-squares residual-step eta,1000 originalvalidation X times8 Gaussian corruptions; no LID fit, no P/gamma combination",
        selection="Each fixed residual-blend R/F selects lambda by original1000cleanvalidation MAE; smallest exact ties, frozen256 testquery audit",
        noisy_test="IndependentGaussian corruptions on original1000test X, same20261412 bank order as prior calibration; actual sampledclean-target MSE, not oracle risk",
        scope="Posthoc investigation, not prospectivelyblinded or label-free scale selection; negativeeta is extrapolation, not partialiteration",
    ),indent=2)+"\n")
    print(json.dumps(choices,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())

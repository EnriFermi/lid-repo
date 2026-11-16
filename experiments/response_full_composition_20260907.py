"""Fixed two-pass canonical denoising: exact chain rule and bias accounting.

No iteration/scale/model selection is performed. g_lambda(y)=b_lambda(b_lambda(y))
is a different field, not the original finite-scale Gaussian posterior. Both
Jacobian factors are evaluated at their actual inputs; J(x)^2 is only a local
prediction whose discrepancy is saved. Correction is anchored at ORIGINAL x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from models.training import diffusion_denoiser_output


ROOT = Path("artifacts/response_full_20260907")


def two_pass(predict, queries, scale, batch_size=16):
    if scale <= 0:
        raise ValueError("Positive physical noise scale required")
    dimension = queries.shape[-1]
    b1, j1 = active_jacobian(predict, queries, dimension, batch_size)
    b2, j2 = active_jacobian(predict, b1, dimension, batch_size)
    jg = j2 @ j1
    offset = b1 - queries
    linear_offset = offset + (j1 @ offset[..., None])[..., 0]
    c1 = offset.square().sum(-1) / scale**2
    c2 = (b2 - queries).square().sum(-1) / scale**2
    r1, r2 = j1.diagonal(dim1=-2, dim2=-1).sum(-1), jg.diagonal(dim1=-2, dim2=-1).sum(-1)
    return {
        "base_posterior": b1, "posterior": b2, "base_jacobian": j1,
        "second_jacobian_at_first_output": j2, "jacobian": jg,
        "base_response": r1, "response": r2, "base_correction": c1,
        "correction": c2, "base_full": r1+c1, "full": r2+c2,
        "local_squared_jacobian": j1 @ j1,
        "jacobian_local_prediction_error_squared": (jg-j1@j1).square().sum((-1, -2)),
        "local_predicted_offset": linear_offset,
        "offset_local_prediction_error_over_lambda_squared":
            (b2-queries-linear_offset).square().sum(-1)/scale**2,
        "local_predicted_response": (j1@j1).diagonal(dim1=-2, dim2=-1).sum(-1),
        "local_predicted_correction": linear_offset.square().sum(-1)/scale**2,
    }


def cases():
    result = []
    for law, dataset, oracle in [
        ("spaghetti", "e8_spaghetti_pca", "population_oracle_ve_spaghetti256/arrays.npz"),
        ("exp", "e6_exp_pca", "population_oracle_exp_fixture/fixture_arrays.npz"),
        ("spiral", "e1_spiral_pca", "population_oracle_spiral_fixture_v2/fixture_arrays.npz"),
    ]:
        for kind in ("posterior", "noise"):
            result.append(dict(name=f"vp_{law}_{kind}", dataset=dataset, family="vp", kind=kind,
                checkpoint=ROOT/f"vp_{law}_paired/final.pt",
                arrays=ROOT/f"vp_{law}_span/{kind}_average_arrays.npz", oracle=ROOT/oracle))
    for arm in ("control", "clean_anchor"):
        result.append(dict(name=f"ve_spaghetti_{arm}", dataset="e8_spaghetti_pca", family="ve", kind=arm,
            checkpoint=ROOT/"clean_anchor_spaghetti/final.pt",
            arrays=ROOT/f"clean_anchor_spaghetti/{arm}_average_arrays.npz",
            oracle=ROOT/"population_oracle_ve_spaghetti256/arrays.npz"))
    return result


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path("models/neural_fields.py"), Path("models/training.py")]
    source_paths += [Path("experiments")/name for name in [
        "response_full_models_20260907_span.py", "response_full_models_20260907_spaghetti_local.py",
        "response_full_models_20260907_ve_refinement.py", "response_full_models_20260907_vp.py",
        "response_full_training_20260907.py"]]
    hashes = {str(p): sha256(p) for p in source_paths}
    rows, replays = [], []
    for case in cases():
        for key in ("checkpoint", "arrays", "oracle"):
            hashes[str(case[key])] = sha256(case[key])
        base, oracle = np.load(case["arrays"]), np.load(case["oracle"])
        for key in ("queries", "query_ids"):
            np.testing.assert_array_equal(base[key], oracle[key])
        np.testing.assert_allclose(base["scales"], oracle["scales"], rtol=1e-9, atol=0)
        q = torch.from_numpy(base["queries"]).double().to(args.device)
        if case["family"] == "vp":
            payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
            field = load_field(payload, case["kind"], "average", args.device)
            predict = lambda x, s: posterior(field, x, s, case["kind"])
        else:
            field = load_refinement_field(case["checkpoint"], case["kind"], "average", args.device)
            predict = lambda x, s: diffusion_denoiser_output(field, x, s, preconditioning="edm", data_std=1.)
        values = {}
        for i, scale in enumerate(base["scales"]):
            scale = float(scale)
            function = lambda x: predict(x, scale)
            tensors = two_pass(function, q, scale, args.batch_size)
            data = {k: v.cpu().numpy() for k, v in tensors.items()}
            for key, prior in [("base_posterior", "posterior"), ("base_response", "response"),
                               ("base_correction", "correction_original")]:
                np.testing.assert_allclose(data[key], base[prior][i], rtol=1e-9, atol=1e-9)
            direct_b, direct_j = active_jacobian(lambda x: function(function(x)), q[:8], q.shape[-1])
            chain_error = float((direct_j-tensors["jacobian"][:8]).abs().max())
            torch.testing.assert_close(direct_b, tensors["posterior"][:8], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(direct_j, tensors["jacobian"][:8], rtol=1e-10, atol=1e-10)
            replays.append(dict(case=case["name"], scale=scale, chain_rule_max_error=chain_error))
            for key, value in data.items():
                values.setdefault(key, []).append(value)
            d = 2 if case["dataset"] == "e6_exp_pca" else 1
            row = dict(case=case["name"], dataset=case["dataset"], scale=scale, count=len(q))
            for prefix in ("base_", ""):
                tag = "base" if prefix else "composed"
                row[tag+"_response_mean"] = float(data[prefix+"response"].mean())
                row[tag+"_correction_mean"] = float(data[prefix+"correction"].mean())
                row[tag+"_value_oracle_error_over_lambda_squared"] = float(
                    np.square(data[prefix+"posterior"]-oracle["posterior"][i]).sum(-1).mean()/scale**2)
                for readout in ("response", "full"):
                    row[tag+"_"+readout+"_d_mae"] = float(np.abs(data[prefix+readout]-d).mean())
                    row[tag+"_"+readout+"_oracle_mae"] = float(np.abs(data[prefix+readout]-oracle[readout][i]).mean())
            row["response_local_prediction_mae"] = float(np.abs(data["response"]-data["local_predicted_response"]).mean())
            row["correction_local_prediction_mae"] = float(np.abs(data["correction"]-data["local_predicted_correction"]).mean())
            for key in ("jacobian_local_prediction_error_squared", "offset_local_prediction_error_over_lambda_squared"):
                row[key] = float(data[key].mean())
            rows.append(row)
            print(json.dumps(row), flush=True)
        np.savez_compressed(args.output/(case["name"]+".npz"),
            queries=base["queries"], query_ids=base["query_ids"], scales=base["scales"],
            **{k: np.stack(v) for k, v in values.items()})
    assert all(sha256(p) == digest for p, digest in hashes.items())
    (args.output/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(dict(
        sources=hashes, source_unchanged=True, completed=True, replays=replays,
        contract="Fixed two-pass same-scale canonical b(b(x)); correction anchored at original x; no LID fit or iteration/scale selection",
        cases=[{k: str(v) if isinstance(v, Path) else v for k,v in c.items()} for c in cases()],
        dtype="float64; exact ordinary-autograd 30x30 Jacobians",
        oracle_scope="Original continuous-law b* used as reference; composed field is not its posterior",
    ), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())

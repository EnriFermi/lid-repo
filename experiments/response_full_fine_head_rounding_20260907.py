"""Fixed float32-head sensitivity check; no second fit or full-float32 claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_convex_head import replaced_head, body_hash
from experiments.response_full_models_20260907_convex_head_loader import load_convex_head
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from models.training import diffusion_denoiser_output


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.parent / "models.pt", args.parent / "fine_ols_arrays.npz",
             args.parent / "fine_ols_noisy.npz", args.parent / "least_squares.npz",
             Path("experiments/response_full_models_20260907_convex_head.py"),
             Path("experiments/response_full_models_20260907_convex_head_loader.py"),
             Path("experiments/response_full_models_20260907_spaghetti_local.py"),
             Path("models/training.py"), Path("models/neural_fields.py")]
    hashes = {str(path): sha256(path) for path in paths}
    model = load_convex_head(args.parent / "models.pt", "fine_ols", args.device).requires_grad_(False)
    coefficient = torch.from_numpy(np.load(args.parent / "least_squares.npz")["coefficient"])
    rounded = replaced_head(model, coefficient, round_head=True).requires_grad_(False)
    assert body_hash(model) == body_hash(rounded)
    saved = np.load(args.parent / "fine_ols_arrays.npz")
    q = torch.from_numpy(saved["queries"]).to(args.device)
    rows, differences = [], {}
    for i, scale in enumerate(saved["scales"]):
        predict = lambda field, x: diffusion_denoiser_output(field, x, float(scale), preconditioning="edm", data_std=1.)
        base, base_j = active_jacobian(lambda x: predict(model, x), q, active=30)
        value, jac = active_jacobian(lambda x: predict(rounded, x), q, active=30)
        base, base_j, value, jac = [x.cpu().numpy() for x in (base, base_j, value, jac)]
        np.testing.assert_allclose(base, saved["posterior"][i], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(base_j, saved["jacobian"][i], rtol=1e-12, atol=1e-12)
        response = np.trace(jac, axis1=-2, axis2=-1)
        correction = ((value-saved["queries"])**2).sum(-1)/scale**2
        current = {"posterior": value-base, "jacobian": jac-base_j,
                   "response": response-saved["response"][i],
                   "correction": correction-saved["correction_original"][i],
                   "full": response+correction-saved["full"][i]}
        for key, array in current.items():
            differences.setdefault(key, []).append(array)
        rows.append({"lambda": float(scale), **{key+"_max_abs": float(abs(array).max())
                                              for key, array in current.items()}})
    noisy = np.load(args.parent / "fine_ols_noisy.npz")
    noisy_rows = []
    for i, scale in enumerate(noisy["scales"]):
        y = torch.from_numpy(noisy["queries"]+float(scale)*noisy["noise"]).to(args.device)
        with torch.no_grad():
            b = diffusion_denoiser_output(rounded, y, float(scale), preconditioning="edm", data_std=1.).cpu().numpy()
        delta = b-noisy["posterior"][i]
        noisy_rows.append({"lambda": float(scale), "posterior_max_abs": float(abs(delta).max()),
                           "change_energy_over_Dlambda2": float((delta**2).mean()/scale**2)})
    np.savez_compressed(args.output / "differences.npz", query_ids=saved["query_ids"], scales=saved["scales"],
                        **{key: np.stack(array) for key, array in differences.items()})
    summary = {"clean": rows, "noisy": noisy_rows,
               "coefficient_max_abs_change": float((coefficient-coefficient.float().double()).abs().max())}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    assert all(sha256(path) == digest for path, digest in hashes.items())
    (args.output / "manifest.json").write_text(json.dumps({"sources": hashes, "completed": True,
        "sources_unchanged": True, "frozen_body_sha256": body_hash(model),
        "scope": "Only fitted head coefficients rounded tofloat32 then inferred infloat64; no new fit, no nonlinear change, not fullfloat32 execution"}, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("artifacts/response_full_20260907/fine_head_spaghetti"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())

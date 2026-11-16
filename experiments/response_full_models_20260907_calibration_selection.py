"""Validation-scale audit of six already-fitted VP displacement calibrations.

Freeze models, training-span projectors and both historical gamma arrays.
The original1000 validation X were used with noise for gamma fitting; now
their known dimension selects lambda separately for each fixed variant/R/F.
This is supervised operating-point selection, not blind or label-free.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_calibration_20260907 import calibrated_response
from experiments.response_full_composition_20260907 import ROOT, cases
from experiments.response_full_composition_selection_20260907 import select_scale
from experiments.response_full_models_20260907_clean_anchor import write_rows
from experiments.response_full_models_20260907_span import projected_trace, sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_multiscale_20260907 import project
from experiments.response_full_training_20260907 import posterior, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS


VARIANTS = ("original", "projected", "calibrated_original", "calibrated_projected")


def fixed_variants(queries, values, response, projected_response, center, basis,
                   scale, gamma_original, gamma_projected):
    """Apply saved scalar coefficients; no fitting, clipping or scale choice."""
    if scale <= 0:
        raise ValueError("Positive physical scale required")
    rank, dimension = basis.shape[1], queries.shape[1]
    anchor = project(queries, center, basis)
    projected = project(values, center, basis)
    functions = {
        "original": (values, response),
        "projected": (projected, projected_response),
        "calibrated_original": (
            queries + gamma_original*(values-queries),
            calibrated_response(response, gamma_original, dimension)),
        "calibrated_projected": (
            anchor + gamma_projected*(projected-anchor),
            calibrated_response(projected_response, gamma_projected, rank)),
    }
    result = {}
    for name, (value, trace) in functions.items():
        correction = ((value-queries)/scale).square().sum(-1)
        result[name+"_posterior"] = value
        result[name+"_response"] = trace
        result[name+"_correction"] = correction
        result[name+"_full"] = trace+correction
    return result


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent_path = args.validation/"manifest.json"
    parent = json.loads(parent_path.read_text())
    assert parent["completed"] and parent["sources_unchanged"]
    hashes = dict(parent["sources"])
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    imports = [Path(__file__), parent_path, args.validation/"selection.json"]
    imports += [Path("experiments")/name for name in (
        "response_full_calibration_20260907.py", "response_full_composition_20260907.py",
        "response_full_composition_selection_20260907.py", "response_full_models_20260907_clean_anchor.py",
        "response_full_models_20260907_span.py", "response_full_models_20260907_spaghetti_local.py",
        "response_full_models_20260907_vp.py", "response_full_multiscale_20260907.py",
        "response_full_training_20260907.py", "ve_known_lid_recoverability_campaign.py")]
    hashes.update({str(path): sha256(path) for path in imports})
    parent_selection = json.loads((args.validation/"selection.json").read_text())
    selections, scale_rows, receipts, replays = [], [], [], []
    manifest = dict(completed=False, source_hashes=hashes, records=receipts, replays=replays,
        variants=list(VARIANTS), checkpoint_rule="All six pre-existing VP AVERAGE states; no head or variant selection",
        calibration_rule="Saved unconstrained noisy-denoising gamma at each scale, no refit or clipping; frozen training-span P",
        selection_rule="Each fixed variant and R/F independently selects lambda by original1000 clean-validation MAE; smallest exact tie",
        validation_reuse="The same1000 validation X were used with8 Gaussian corruptions per X to fit gamma; lambda selection now uses known d explicitly",
        scope="Supervised benchmark-tuned posthoc operating-point audit, not blind, label-free, or a new training campaign",
        test_rule="Original untouched256 test query IDs; no coefficients, head, variant or scale selected from test",
        trace="Exact tr(PJ) using complete fixed span basis; full validation b/R reused with independent first8 full-J gate",
        batch_size=args.batch_size)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for case in [item for item in cases() if item["family"] == "vp"]:
        calibration_dir = ROOT/("calibration_"+case["name"])
        calibration_path = calibration_dir/"arrays.npz"
        calibration_manifest_path = calibration_dir/"manifest.json"
        validation_path = args.validation/(case["name"]+".npz")
        fixture_path = case["checkpoint"].parent/"fixture.pt"
        calibration_manifest = json.loads(calibration_manifest_path.read_text())
        config = calibration_manifest["args"]
        assert config["family"] == "vp" and config["mode"] == "average"
        assert config["kind"] == case["kind"] and config["dataset"] == case["dataset"]
        assert Path(config["checkpoint"]) == case["checkpoint"]
        assert Path(config["arrays"]) == case["arrays"]
        assert calibration_manifest["fit_replicates"] == 8
        assert sha256(case["checkpoint"]) == calibration_manifest["checkpoint_sha256"]
        assert sha256(case["arrays"]) == calibration_manifest["arrays_sha256"]
        assert sha256("experiments/response_full_calibration_20260907.py") == calibration_manifest["source_sha256"]
        for path in (calibration_path, calibration_manifest_path, calibration_dir/"summary.json",
                     validation_path, fixture_path, case["checkpoint"], case["arrays"]):
            hashes[str(path)] = sha256(path)
        base, calibration, validation = [np.load(path) for path in
            (case["arrays"], calibration_path, validation_path)]
        payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        for key in ("queries", "query_ids", "scales"):
            np.testing.assert_array_equal(calibration[key], base[key])
        np.testing.assert_array_equal(validation["scales"], base["scales"])
        np.testing.assert_array_equal(validation["test_query_ids"], base["query_ids"])
        np.testing.assert_array_equal(base["queries"], fixture["queries"].numpy())
        np.testing.assert_array_equal(base["normalization_mean"], fixture["mean"].numpy())
        assert float(base["normalization_rms"]) == float(fixture["rms"])
        normalized = (torch.from_numpy(validation["validation_raw"])-fixture["mean"])/float(fixture["rms"])
        np.testing.assert_array_equal(normalized.numpy(), validation["validation_queries"])
        assert len(normalized) == 1000
        assert tensor_sha(torch.from_numpy(validation["validation_raw"])) == fixture["split_hashes"]["selection"]
        np.testing.assert_array_equal(validation["validation_ids"], np.arange(1000))
        np.testing.assert_array_equal(validation["validation_global_rows"], payload["config"]["train_count"]+np.arange(1000))
        for name in ("original_gamma", "projected_gamma"):
            assert calibration[name].shape == (9,) and np.isfinite(calibration[name]).all()
        scales = base["scales"]
        qval = normalized.double().to(args.device)
        qtest = torch.from_numpy(base["queries"]).double().to(args.device)
        center = torch.from_numpy(base["span_center"]).double().to(args.device)
        basis = torch.from_numpy(base["span_basis"]).double().to(args.device)
        torch.testing.assert_close(basis.T@basis, torch.eye(basis.shape[1], device=args.device, dtype=torch.float64), atol=1e-10, rtol=1e-10)
        field = load_field(payload, case["kind"], "average", args.device)
        all_values = {}
        for index, scale in enumerate(scales):
            scale = float(scale)
            predict = lambda x: posterior(field, x, scale, case["kind"])
            actual_values, span_r = projected_trace(predict, qval, basis, args.batch_size)
            saved_values = torch.from_numpy(validation["base_posterior"][index]).to(qval)
            saved_r = torch.from_numpy(validation["base_response"][index]).to(qval)
            torch.testing.assert_close(actual_values, saved_values, rtol=1e-9, atol=1e-9)
            _, direct_j = active_jacobian(predict, qval[:8], qval.shape[1], 8)
            direct_r = direct_j.diagonal(dim1=-2, dim2=-1).sum(-1)
            direct_span = (basis.T@direct_j@basis).diagonal(dim1=-2, dim2=-1).sum(-1)
            torch.testing.assert_close(direct_r, saved_r[:8], rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(direct_span, span_r[:8], rtol=1e-9, atol=1e-9)
            gammas = [float(calibration[name][index]) for name in ("original_gamma", "projected_gamma")]
            val = fixed_variants(qval, saved_values, saved_r, span_r, center, basis, scale, *gammas)
            test = fixed_variants(qtest, torch.from_numpy(base["posterior"][index]).to(qtest),
                torch.from_numpy(base["response"][index]).to(qtest),
                torch.from_numpy(base["response_span"][index]).to(qtest), center, basis, scale, *gammas)
            calibration_replay = 0.
            for variant, original_key in (("calibrated_original", "original"), ("calibrated_projected", "projected")):
                for readout in ("posterior", "response", "correction", "full"):
                    actual = test[variant+"_"+readout].cpu().numpy()
                    saved = calibration[original_key+"_"+readout][index]
                    np.testing.assert_allclose(actual, saved, rtol=1e-9, atol=1e-9)
                    calibration_replay = max(calibration_replay, float(np.max(abs(actual-saved))))
            for key, saved_key in (("projected_posterior", "posterior_projected"),
                                   ("projected_response", "response_span"), ("projected_full", "full_projected")):
                np.testing.assert_allclose(test[key].cpu().numpy(), base[saved_key][index], rtol=1e-9, atol=1e-9)
            for split, values in (("validation", val), ("test", test)):
                for key, value in values.items():
                    all_values.setdefault(split+"_"+key, []).append(value.cpu().numpy())
            replays.append(dict(case=case["name"], scale=scale,
                validation_b_max=float((actual_values-saved_values).abs().max()),
                validation_R_max=float((direct_r-saved_r[:8]).abs().max()),
                validation_projected_R_max=float((direct_span-span_r[:8]).abs().max()),
                frozen_calibration_readouts_max=calibration_replay))
            print(json.dumps(dict(case=case["name"], scale=scale, projected_trace_count=1000)), flush=True)
        arrays = {key: np.stack(value) for key, value in all_values.items()}
        truth = DATASET_SPECS[case["dataset"]].true_lid
        selected_arrays = {}
        for variant in VARIANTS:
            for readout in ("response", "full"):
                key = variant+"_"+readout
                selected, risks = select_scale(arrays["validation_"+key], truth, scales)
                test_values = arrays["test_"+key][selected]
                row = dict(case=case["name"], dataset=case["dataset"], variant=variant,
                    readout=readout, true_dimension=truth, validation_count=1000, test_count=len(qtest),
                    selected_index=selected, selected_scale=float(scales[selected]),
                    at_grid_boundary=bool(selected in (0, len(scales)-1)),
                    validation_mae=float(risks[selected]), test_mae=float(np.abs(test_values-truth).mean()),
                    all_scale_validation_mae=risks.tolist())
                if variant == "original":
                    prior = [r for r in parent_selection if r["case"] == case["name"] and r["passes"] == 1 and r["readout"] == readout][0]
                    assert row["selected_index"] == prior["selected_index"]
                    np.testing.assert_allclose([row["validation_mae"], row["test_mae"]], [prior["validation_mae"], prior["test_mae"]], rtol=1e-9, atol=1e-9)
                selections.append(row)
                selected_arrays["test_selected_"+key] = test_values
                for i, scale in enumerate(scales):
                    scale_rows.append(dict(case=case["name"], variant=variant, readout=readout,
                        scale=float(scale), validation_mae=float(risks[i]),
                        test_mae=float(np.abs(arrays["test_"+key][i]-truth).mean()),
                        selected=bool(i == selected)))
        np.savez_compressed(args.output/(case["name"]+".npz"),
            validation_queries=validation["validation_queries"], validation_raw=validation["validation_raw"],
            validation_ids=validation["validation_ids"], validation_global_rows=validation["validation_global_rows"],
            test_queries=base["queries"], test_query_ids=base["query_ids"], scales=scales,
            normalization_mean=base["normalization_mean"], normalization_rms=base["normalization_rms"],
            span_center=base["span_center"], span_basis=base["span_basis"],
            original_gamma=calibration["original_gamma"], projected_gamma=calibration["projected_gamma"],
            **arrays, **selected_arrays)
        receipts.append(dict(case=case["name"], calibration_manifest=str(calibration_manifest_path),
            span_rank=basis.shape[1], true_dimension=truth, validation_raw_and_normalization_bitwise=True,
            original_selection_replayed=True, coefficients_refitted=False,
            original_gamma=calibration["original_gamma"].tolist(), projected_gamma=calibration["projected_gamma"].tolist()))
        write_rows(args.output, "selection", selections)
        write_rows(args.output, "all_scales", scale_rows)
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    assert len(selections) == 48 and len(scale_rows) == 432
    assert all(sha256(path) == digest for path, digest in hashes.items())
    manifest.update(completed=True, sources_unchanged=True, elapsed_seconds=time.monotonic()-started)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(selections, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=ROOT/"two_pass_validation_selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())

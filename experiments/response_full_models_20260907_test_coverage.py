"""Extend36 frozen selected VP readouts from256 to all1000 originaltest X.

Only the union of already-selected scales is evaluated. No fit, scale choice,
head/variant selection, new noise or new data draw is performed. The additional
744 clean-query evaluations are coverage, not prospectively blinded testing:
all1000 X already appeared in the old independent noisy-test risk bank.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_composition_20260907 import ROOT, cases, two_pass
from experiments.response_full_models_20260907_clean_anchor import write_rows
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_residual_calibration_20260907 import blend
from experiments.response_full_training_20260907 import posterior, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits


def frozen_choices(composition, residual):
    result = []
    for row in composition:
        if row["passes"] not in (1, 2):
            raise ValueError("Only frozen K1/K2 choices are supported")
        result.append({**row, "operator":"one_pass" if row["passes"] == 1 else "two_pass"})
    for row in residual:
        if not np.isfinite(row["eta"]):
            raise ValueError("Frozen coefficient must be finite")
        result.append({**row, "operator":"residual"})
    identities = [(r["case"], r["operator"], r["readout"]) for r in result]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate frozen estimator/readout")
    return result


def coverage_indices(old_ids, count):
    ids = np.asarray(old_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Integer one-dimensional original query IDs required")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("Original query IDs must be unique and in range")
    additional = np.setdiff1d(np.arange(count), ids)
    return ids, additional


def coverage_metrics(values, old_ids, dimension):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Finite scalar readout per test point required")
    old, extra = coverage_indices(old_ids, len(values))
    error = np.abs(values-dimension)
    result = dict(old_count=len(old), additional_count=len(extra), full_count=len(values),
        old256_mae=float(error[old].mean()), additional744_mae=float(error[extra].mean()),
        all1000_mae=float(error.mean()))
    weighted = (len(old)*result["old256_mae"]+len(extra)*result["additional744_mae"])/len(values)
    np.testing.assert_allclose(weighted, result["all1000_mae"], rtol=1e-12, atol=1e-12)
    return result


def residual_values(first, second, rfirst, rsecond, query, scale, coefficient):
    value = blend(first, second, coefficient)
    response = blend(rfirst, rsecond, coefficient)
    correction = ((value-query)/scale).square().sum(-1)
    return value, response, correction


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    composition_choice_path = ROOT/"two_pass_validation_selection/selection.json"
    residual_choice_path = ROOT/"residual_step_calibration/selection.json"
    comp = json.loads(composition_choice_path.read_text())
    residual = json.loads(residual_choice_path.read_text())
    choices = frozen_choices(comp, residual)
    assert len(comp) == 24 and len(residual) == 12 and len(choices) == 36
    hashes = {}
    for path in (ROOT/"two_pass_validation_selection/manifest.json", ROOT/"residual_step_calibration/manifest.json"):
        parent = json.loads(path.read_text())
        assert parent["completed"] and parent["sources_unchanged"]
        for name, digest in parent["sources"].items():
            assert sha256(name) == digest, name
            if name in hashes:
                assert hashes[name] == digest
            hashes[name] = digest
        hashes[str(path)] = sha256(path)
    paths = [Path(__file__), composition_choice_path, residual_choice_path]
    paths += [Path("experiments")/name for name in (
        "response_full_composition_20260907.py", "response_full_models_20260907_clean_anchor.py",
        "response_full_models_20260907_span.py", "response_full_models_20260907_spaghetti_local.py",
        "response_full_models_20260907_vp.py", "response_full_residual_calibration_20260907.py",
        "response_full_training_20260907.py", "ve_known_lid_recoverability_campaign.py")]
    hashes.update({str(path):sha256(path) for path in paths})
    rows, replays, split_receipts = [], [], []
    manifest = dict(completed=False, sources=hashes, frozen_choices=choices, replays=replays,
        split_receipts=split_receipts, scope="Coverage extension of36 previously frozen choices; no fitting, choice changes, new noise or data",
        sample_scope="Original1000 test X, not newly generated or prospectivelyblinded; already used in independent noisy-test risk",
        evaluation_rule="Only union of selected lambda within each fixed average model; same canonical function and exact actual two-pass traces",
        groups="Original256 IDs in original order; disjoint remaining744; full1000",
        storage="Values and scalarR/C only; full Jacobians are temporary derivative gates, never archived",
        batch_size=args.batch_size)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    total_scale_cells = 0
    for case in [c for c in cases() if c["family"] == "vp"]:
        source = ROOT/"two_pass_composition"/(case["name"]+".npz")
        residual_source = ROOT/"residual_step_calibration"/(case["name"]+".npz")
        fixture_path = case["checkpoint"].parent/"fixture.pt"
        for path in (source, residual_source, fixture_path, case["checkpoint"]):
            hashes[str(path)] = sha256(path)
        old, old_residual = np.load(source), np.load(residual_source)
        payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        raw_train, raw_val, raw_test = make_splits(case["dataset"], train_size=cfg["train_count"],
            validation_size=cfg["selection_count"], test_size=cfg["test_count"], data_seed=0)
        raw_hashes = {name:tensor_sha(torch.from_numpy(values)) for name,values in
            (("train",raw_train), ("selection",raw_val), ("test",raw_test))}
        assert raw_hashes == fixture["split_hashes"]
        assert len(raw_test) == 1000
        normalized = (torch.from_numpy(raw_test)-fixture["mean"])/float(fixture["rms"])
        np.testing.assert_array_equal(normalized.numpy(), old_residual["noisy_test_clean"])
        ids, extra = coverage_indices(old["query_ids"], 1000)
        assert len(ids) == 256 and len(extra) == 744
        np.testing.assert_array_equal(ids, fixture["query_ids"])
        np.testing.assert_array_equal(ids, old_residual["test_query_ids"])
        np.testing.assert_array_equal(normalized[ids].numpy(), old["queries"])
        np.testing.assert_array_equal(old["queries"], old_residual["test_queries"])
        np.testing.assert_array_equal(old["scales"], old_residual["scales"])
        selected = [row for row in choices if row["case"] == case["name"]]
        assert len(selected) == 6
        indices = sorted(set(row["selected_index"] for row in selected))
        for row in selected:
            assert float(old["scales"][row["selected_index"]]) == row["selected_scale"]
            if row["operator"] == "residual":
                assert float(old_residual["eta"][row["selected_index"]]) == row["eta"]
        field = load_field(payload, case["kind"], "average", args.device)
        queries = normalized.double().to(args.device)
        truth = DATASET_SPECS[case["dataset"]].true_lid
        values, selected_values = {}, {}
        for index in indices:
            scale = float(old["scales"][index])
            predict = lambda x:posterior(field, x, scale, case["kind"])
            result = two_pass(predict, queries, scale, args.batch_size)
            _, direct = active_jacobian(lambda x:predict(predict(x)), queries[:8], queries.shape[1], 8)
            torch.testing.assert_close(direct, result["jacobian"][:8], rtol=1e-10, atol=1e-10)
            baseline_replay = 0.
            kept = ("base_posterior", "posterior", "base_response", "response", "base_correction", "correction")
            for key in kept:
                current = result[key][ids].cpu().numpy()
                np.testing.assert_allclose(current, old[key][index], rtol=1e-9, atol=1e-9)
                baseline_replay = max(baseline_replay, float(np.max(abs(current-old[key][index]))))
                values.setdefault(key, []).append(result[key].cpu().numpy())
            coefficient = float(old_residual["eta"][index])
            rb, rr, rc = residual_values(result["base_posterior"], result["posterior"],
                result["base_response"], result["response"], queries, scale, coefficient)
            residual_replay = 0.
            for value, key in ((rb,"test_posterior"),(rr,"test_response"),(rc,"test_correction")):
                current = value[ids].cpu().numpy()
                np.testing.assert_allclose(current, old_residual[key][index], rtol=1e-9, atol=1e-9)
                residual_replay = max(residual_replay, float(np.max(abs(current-old_residual[key][index]))))
            for choice in [row for row in selected if row["selected_index"] == index]:
                if choice["operator"] == "residual":
                    r, c = rr, rc
                else:
                    prefix = "base_" if choice["operator"] == "one_pass" else ""
                    r, c = result[prefix+"response"], result[prefix+"correction"]
                scalar = r if choice["readout"] == "response" else r+c
                scalar = scalar.cpu().numpy()
                metrics = coverage_metrics(scalar, ids, truth)
                np.testing.assert_allclose(metrics["old256_mae"], choice["test_mae"], rtol=1e-9, atol=1e-9)
                row = dict(case=case["name"], dataset=case["dataset"], operator=choice["operator"],
                    readout=choice["readout"], true_dimension=truth, selected_index=index, selected_scale=scale,
                    eta=choice.get("eta"), at_grid_boundary=choice["at_grid_boundary"],
                    frozen_validation_mae=choice["validation_mae"], original_reported256_mae=choice["test_mae"], **metrics)
                rows.append(row)
                selected_values[choice["operator"]+"_"+choice["readout"]] = scalar
            replays.append(dict(case=case["name"], selected_scale=scale,
                old256_one_two_b_R_C_max=baseline_replay, old256_residual_b_R_C_max=residual_replay,
                direct_two_pass_jacobian_max=float((direct-result["jacobian"][:8]).abs().max())))
            total_scale_cells += 1
            print(json.dumps(dict(case=case["name"], selected_scale=scale, test_count=1000)), flush=True)
        np.savez_compressed(args.output/(case["name"]+".npz"), test_queries=normalized.numpy(),
            test_query_ids=np.arange(1000), original_query_ids=ids, additional_query_ids=extra,
            source_scale_indices=np.asarray(indices), scales=old["scales"][indices],
            frozen_eta_at_evaluated_scales=old_residual["eta"][indices],
            normalization_mean=fixture["mean"].numpy(), normalization_rms=float(fixture["rms"]),
            **{key:np.stack(value) for key,value in values.items()}, **selected_values)
        split_receipts.append(dict(case=case["name"], split_hashes=raw_hashes,
            original256_queries_bitwise=True, original1000_noisy_bank_clean_X_bitwise=True,
            count=1000, old_count=256, additional_count=744, source_scale_indices=indices))
        write_rows(args.output, "coverage", rows)
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    assert len(rows) == 36
    assert all(sha256(path) == digest for path,digest in hashes.items())
    manifest.update(completed=True, sources_unchanged=True, evaluated_model_scale_cells=total_scale_cells,
                    elapsed_seconds=time.monotonic()-started)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())

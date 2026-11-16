"""Supervised validation-scale evaluation of fixed one/two-pass VP readouts.

Restore the original1000 selection examples, not test points. Choose each
operator/readout's scale by validation MAE, with the first (smallest) scale
winning exact ties. Number of passes and head are never selected. This is the
known-LID benchmark-tuned track, not a deployable label-free selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_composition_20260907 import ROOT, cases, two_pass
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits


def select_scale(validation_values, true_dimension, scales):
    values, grid = np.asarray(validation_values), np.asarray(scales)
    if values.ndim != 2 or len(values) != len(grid) or not np.isfinite(values).all():
        raise ValueError("Finite scale-by-validation-query values required")
    if not (np.diff(grid) > 0).all():
        raise ValueError("Strictly increasing scale grid required")
    risk = np.abs(values-true_dimension).mean(axis=1)
    return int(np.argmin(risk)), risk


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path("experiments/response_full_composition_20260907.py"),
        Path("experiments/response_full_training_20260907.py"),
        Path("experiments/response_full_models_20260907_vp.py"),
        Path("experiments/response_full_models_20260907_spaghetti_local.py"),
        Path("experiments/response_full_models_20260907_span.py"),
        Path("experiments/ve_known_lid_recoverability_campaign.py"),
        Path("models/neural_fields.py"), ROOT/"two_pass_composition/manifest.json"]
    hashes = {str(p): sha256(p) for p in paths}
    rows, replays, split_receipts = [], [], []
    for case in [c for c in cases() if c["family"] == "vp"]:
        fixture_path = case["checkpoint"].parent/"fixture.pt"
        test_path = ROOT/"two_pass_composition"/(case["name"]+".npz")
        for path in [fixture_path, case["checkpoint"], test_path]:
            hashes[str(path)] = sha256(path)
        payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        raw_train, raw_val, raw_test = make_splits(case["dataset"],
            train_size=cfg["train_count"], validation_size=cfg["selection_count"],
            test_size=cfg["test_count"], data_seed=0)
        raw_hashes = {k: tensor_sha(torch.from_numpy(x)) for k,x in
                      [("train", raw_train), ("selection", raw_val), ("test", raw_test)]}
        assert raw_hashes == fixture["split_hashes"]
        assert len(raw_val) == 1000
        mean, rms = fixture["mean"], float(fixture["rms"])
        validation = (torch.from_numpy(raw_val)-mean)/rms
        torch.testing.assert_close(validation[:len(fixture["validation"])], fixture["validation"], rtol=0, atol=0)
        test = np.load(test_path)
        np.testing.assert_array_equal(test["query_ids"], fixture["query_ids"])
        test_restored = ((torch.from_numpy(raw_test[test["query_ids"]])-mean)/rms).numpy()
        np.testing.assert_array_equal(test["queries"], test_restored)
        field = load_field(payload, case["kind"], "average", args.device)
        q = validation.double().to(args.device)
        scales = test["scales"]
        values = {}
        for i, s in enumerate(scales):
            function = lambda x: posterior(field, x, float(s), case["kind"])
            result = two_pass(function, q, float(s), args.batch_size)
            _, direct = active_jacobian(lambda x: function(function(x)), q[:8], q.shape[-1])
            torch.testing.assert_close(direct, result["jacobian"][:8], rtol=1e-10, atol=1e-10)
            replays.append(dict(case=case["name"], scale=float(s),
                direct_chain_max_error=float((direct-result["jacobian"][:8]).abs().max())))
            for key in ["base_posterior", "posterior", "base_response", "response",
                        "base_correction", "correction", "base_full", "full"]:
                values.setdefault(key, []).append(result[key].cpu().numpy())
            print(json.dumps(dict(case=case["name"], scale=float(s), validation_count=len(q))), flush=True)
        arrays = {k: np.stack(v) for k,v in values.items()}
        truth = DATASET_SPECS[case["dataset"]].true_lid
        for prefix, passes in [("base_", 1), ("", 2)]:
            for readout in ("response", "full"):
                selected, risk = select_scale(arrays[prefix+readout], truth, scales)
                test_value = test[prefix+readout][selected]
                rows.append(dict(case=case["name"], dataset=case["dataset"], passes=passes,
                    readout=readout, true_dimension=truth, validation_count=len(q),
                    test_count=len(test_value), selected_index=selected,
                    selected_scale=float(scales[selected]),
                    at_grid_boundary=bool(selected in [0, len(scales)-1]),
                    validation_mae=float(risk[selected]),
                    test_mae=float(np.abs(test_value-truth).mean()),
                    all_scale_validation_mae=risk.tolist()))
        np.savez_compressed(args.output/(case["name"]+".npz"),
            validation_raw=raw_val, validation_queries=validation.numpy(),
            validation_ids=np.arange(len(raw_val)),
            validation_global_rows=cfg["train_count"]+np.arange(len(raw_val)),
            test_query_ids=test["query_ids"], scales=scales, **arrays)
        split_receipts.append(dict(case=case["name"], restored_raw_hashes=raw_hashes,
            validation_prefix_bitwise=True, test_queries_bitwise=True,
            normalization_mean=tensor_sha(mean), normalization_rms=rms))
    assert all(sha256(p) == digest for p,digest in hashes.items())
    (args.output/"selection.json").write_text(json.dumps(rows, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(dict(
        sources=hashes, completed=True, sources_unchanged=True,
        restored_splits=split_receipts, direct_chain_replays=replays,
        contract="All six fixed VP averages, K=1/2 fixed; each R/F selects lambda ONLY by original1000 validation MAE; smallest exact-tie scale; untouched256 test subset",
        selection_scope="Known-LID benchmark-tuned coefficient-space diagnostic, not label-free and not a full historical rendered benchmark run",
        true_lid_note="Spaghetti has one latent phase and d=1, not upstream training label20",
    ), indent=2)+"\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())

"""Fixed-state evaluation of the paired noise-energy continuation.

Clean validation chooses only physical scale, separately for response and full.
No state, checkpoint, regularizer weight, or test-selected scale is selected.
Population oracles are reused only at their exact original scales and queries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.response_full_training_20260907 import posterior, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import make_splits
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


ROOT = Path("artifacts/response_full_20260907")
ORACLES = {
    "e6_exp_pca": ROOT / "population_oracle_exp_fixture/fixture_arrays.npz",
    "e1_spiral_pca": ROOT / "population_oracle_spiral_fixture_v2/fixture_arrays.npz",
}
DIMENSIONS = {"e6_exp_pca": 2, "e1_spiral_pca": 1}
STATES = ("initial", "control_last", "moment_last", "control_average", "moment_average")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physical_grid(anchors):
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.shape != (9,) or not np.all(np.diff(anchors) > 0):
        raise ValueError("Nine increasing oracle scales required")
    grid = np.geomspace(0.01, 1.0, 25)
    if not np.allclose(grid[::3], anchors, rtol=1e-8, atol=0):
        raise ValueError("Oracle anchors do not match the fixed physical grid")
    grid[::3] = anchors
    return grid


def select_scale(validation_values, dimension, scales):
    values, scales = np.asarray(validation_values), np.asarray(scales)
    if values.ndim != 2 or values.shape[0] != len(scales) or not np.isfinite(values).all():
        raise ValueError("Finite scale-by-validation-query values required")
    if not np.all(np.diff(scales) > 0):
        raise ValueError("Scales must be increasing")
    risks = np.abs(values - dimension).mean(axis=1)
    return int(np.argmin(risks)), risks


def load_model(payload, state, device):
    """Continuation tensors are double: convert BEFORE state_dict loading."""
    if state == "initial":
        tensors = payload["models"]["control"]
    else:
        arm, mode = state.split("_")
        tensors = payload["averages" if mode == "average" else "models"][arm]
    model = ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    model = model.double()
    model.load_state_dict(tensors, strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, tensors[key].double(), rtol=0, atol=0)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def readouts(model, queries, scale, batch_size):
    result = {"posterior": [], "response": [], "correction": []}
    with torch.no_grad():
        for q in queries.split(batch_size):
            b, r = fast_value_and_trace(model, q, float(scale), "vp_trigonometric", output_coordinate="data")
            result["posterior"].append(b.cpu().numpy())
            result["response"].append(r.cpu().numpy())
            result["correction"].append(((b-q).square().sum(-1)/float(scale)**2).cpu().numpy())
    output = {key: np.concatenate(value) for key, value in result.items()}
    output["full"] = output["response"] + output["correction"]
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("Nonfinite learned readout")
    return output


def noisy_metrics(predicted, response, clean, noise, scale):
    error = predicted - clean
    dim = clean.shape[-1]
    original_per_query = np.square(error).sum(-1)/(dim*scale**2)
    # Algebraically F(Y)-||Z||^2, avoiding subtraction of large Y-based terms.
    residual = response + np.square(error).sum(-1)/scale**2 - 2*(noise*error).sum(-1)/scale
    target = np.square(noise).sum(-1)
    direct = response + np.square(predicted-clean-scale*noise).sum(-1)/scale**2 - target
    np.testing.assert_allclose(residual, direct, rtol=1e-9, atol=1e-8)
    return {"original_denoising_loss": float(original_per_query.mean()),
            "moment_mse_unscaled": float(np.square(residual).mean()),
            "moment_mse_div_D": float(np.square(residual).mean()/dim),
            "mean_noise_energy": float(target.mean()),
            "mean_noisy_full": float((residual+target).mean()),
            "moment_residual_mean": float(residual.mean())}


def active_span_trace(model, queries, scale, active_dimension, batch_size):
    """Exact diagonal sum in the known global active span, not local tangent."""
    if not 0 < active_dimension <= queries.shape[-1]:
        raise ValueError("Invalid active-span dimension")
    values = []
    for block in queries.split(batch_size):
        q = block.detach().clone().requires_grad_(True)
        b = posterior(model, q, float(scale), "posterior")
        trace = sum(torch.autograd.grad(b[:,d].sum(), q, retain_graph=d<active_dimension-1)[0][:,d]
                    for d in range(active_dimension))
        values.append(trace.detach().cpu().numpy())
    return np.concatenate(values)


def check_oracle(fixture, oracle, test_queries):
    np.testing.assert_array_equal(oracle["query_ids"], fixture["query_ids"])
    np.testing.assert_array_equal(oracle["queries"], fixture["queries"].numpy())
    np.testing.assert_array_equal(oracle["queries"], test_queries[oracle["query_ids"]])
    np.testing.assert_array_equal(oracle["normalization_mean"], fixture["mean"].numpy())
    assert float(oracle["normalization_scale"]) == float(fixture["rms"])


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.noise_replicates < 1 or args.batch_size < 1:
        raise ValueError("Positive batch and replica counts required")
    args.output.mkdir(parents=True, exist_ok=False)
    initial_path, final_path = args.source/"initial.pt", args.source/"final.pt"
    initial = torch.load(initial_path, weights_only=False, map_location="cpu")
    final = torch.load(final_path, weights_only=False, map_location="cpu")
    training_manifest = json.loads((args.source/"manifest.json").read_text())
    assert training_manifest["completed"] and training_manifest["all_sources_unchanged"]
    assert initial["schema"] == final["schema"] == "response_full_moment_continuation_v1"
    assert initial["continuation_step"] == 0
    assert final["continuation_step"] == final["config"]["steps"]
    assert final["average_count"] == final["config"]["average_steps"]
    assert initial["architecture"] == final["architecture"]
    assert initial["config"] == final["config"]
    gamma = float(final["config"]["gamma"])
    source_checkpoint = Path(final["source_checkpoint"])
    source_fixture = Path(final["source_fixture"])
    parent = torch.load(source_checkpoint, weights_only=False, map_location="cpu")
    fixture = torch.load(source_fixture, weights_only=False, map_location="cpu")
    dataset = parent["config"]["dataset"]
    active_dimension = 3 if dataset == "e6_exp_pca" else 2
    oracle_path = ORACLES[dataset]
    paths = [Path(__file__), initial_path, final_path, args.source/"manifest.json", args.source/"history.json",
             source_checkpoint, source_fixture, source_checkpoint.parent/"manifest.json",
             source_checkpoint.parent/"predictions.npz", oracle_path, oracle_path.parent/"summary.json",
             Path("experiments/response_full_training_20260907.py"),
             Path("experiments/response_full_moment_training_20260907.py"),
             Path("experiments/ve_known_lid_recoverability_campaign.py"),
             Path("experiments/fm_mechanism_fast_trace.py"),
             Path("experiments/fm_mechanism_diagnostics.py"), Path("models/neural_fields.py")]
    hashes = {str(path): sha(path) for path in paths}
    history = json.loads((args.source/"history.json").read_text())
    assert len(history) == final["continuation_step"]
    for payload in (initial, final):
        assert payload["source_checkpoint_sha256"] == sha(source_checkpoint)
        assert payload["source_fixture_sha256"] == sha(source_fixture)
    for path, digest in training_manifest["sources"].items():
        assert sha(path) == digest, path
    # Only source LAST is the paired initialization. Historical averages are not substituted.
    for key, value in parent["models"]["posterior"].items():
        for arm in ("control", "moment"):
            torch.testing.assert_close(initial["models"][arm][key], value.double(), rtol=0, atol=0)
    cfg = parent["config"]
    raw_train, raw_val, raw_test = make_splits(dataset, train_size=cfg["train_count"],
        validation_size=cfg["selection_count"], test_size=cfg["test_count"], data_seed=0)
    split_hashes = {name: tensor_sha(torch.from_numpy(raw)) for name, raw in
                   [("train", raw_train), ("selection", raw_val), ("test", raw_test)]}
    assert split_hashes == fixture["split_hashes"]
    assert len(raw_val) == len(raw_test) == 1000
    mean, rms = fixture["mean"], float(fixture["rms"])
    for payload in (initial, final, parent):
        torch.testing.assert_close(payload["normalization_mean"].double(), mean.double(), rtol=0, atol=0)
        assert float(payload["normalization_scale"]) == rms
    # Preserve the original float32 normalization, then evaluate that same input in double.
    val = (torch.from_numpy(raw_val)-mean)/rms
    test = (torch.from_numpy(raw_test)-mean)/rms
    torch.testing.assert_close(val[:len(fixture["validation"])], fixture["validation"], rtol=0, atol=0)
    oracle = np.load(oracle_path)
    check_oracle(fixture, oracle, test.numpy())
    scales = physical_grid(oracle["scales"])
    oracle_indices = np.arange(0, 25, 3)
    assert np.array_equal(scales[oracle_indices], oracle["scales"])
    noise_ids = np.repeat(np.arange(len(val)), args.noise_replicates)
    rng = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    z = torch.randn((len(noise_ids), val.shape[-1]), generator=rng, dtype=torch.float64)
    noisy_clean = val[noise_ids].double()
    np.savez_compressed(args.output/"fixture.npz", validation_raw=raw_val, test_raw=raw_test,
        validation_queries=val.numpy(), test_queries=test.numpy(),
        validation_ids=np.arange(len(val)), test_ids=np.arange(len(test)),
        normalization_mean=mean.numpy(), normalization_scale=rms, scales=scales,
        oracle_query_ids=oracle["query_ids"], oracle_scale_indices=oracle_indices,
        noise_clean_validation_ids=noise_ids, noise=z.numpy())
    manifest = dict(completed=False, source_hashes=hashes, states=list(STATES),
        dataset=dataset, dimension=DIMENSIONS[dataset], ambient_dimension=val.shape[-1],
        split_hashes=split_hashes, original_float32_normalization_preserved=True,
        oracle_query_scale_normalization_match=True, source_initialization="parent models.posterior LAST",
        scale_selection="Each fixed state's R/F chooses scale only by 1000 validation MAE; smallest exact tie",
        label_scope="Known-d benchmark-tuned scale selection, not label-free; training moment uses no d labels",
        checkpoint_selection=False, test_selection=False, noise_seed=args.noise_seed,
        noise_replicates=args.noise_replicates, shared_noise_across_states_and_scales=True,
        oracle_scope="Existing 256 test queries, exactly 9 source scales only; no interpolation or new oracle",
        continuation_config=final["config"], normalization_scale=rms,
        gamma=gamma, moment_normalization="mean residual squared divided by ambient D, matching trainer",
        primary_comparison="control_average versus moment_average; initial/last are fixed diagnostics",
        active_span_diagnostic="Known global coefficient span, 3 Exp / 2 Spiral; not the local tangent space. Exact ordinary-AD diagonal sum at 9 oracle scales x 256 test queries",
        noisy_risk_scope="Original1000 validation clean examples with fixed Gaussian replicas; 25-scale summaries descriptive, not exact population integrals",
        started=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    all_rows, oracle_rows, risk_rows, selections, replays, cache_replays = [], [], [], [], [], []
    for state in STATES:
        model = load_model(initial if state == "initial" else final, state, args.device)
        if state == "initial":
            cached = np.load(source_checkpoint.parent/"predictions.npz")
            for j in (0, 4, 8):
                values = readouts(model, fixture["queries"][:6].double().to(args.device),
                                  cached["scales"][j], args.batch_size)
                maximum = 0.
                for key, cached_key in (("posterior", "b"), ("response", "response"),
                                        ("correction", "correction")):
                    reference = cached["posterior_last_"+cached_key][j,:6]
                    np.testing.assert_allclose(values[key], reference, rtol=1e-9, atol=1e-9)
                    maximum = max(maximum, float(abs(values[key]-reference).max()))
                cache_replays.append(dict(scale=float(cached["scales"][j]), max_error=maximum))
        arrays = {}
        for split, points in (("validation", val), ("test", test)):
            blocks = {key: [] for key in ("posterior", "response", "correction", "full")}
            for scale in scales:
                values = readouts(model, points.double().to(args.device), scale, args.batch_size)
                for key in blocks:
                    blocks[key].append(values[key])
            arrays.update({split+"_"+key: np.stack(value) for key, value in blocks.items()})
        # Ordinary reverse AD replay does not use the fast-trace implementation.
        for j in (0, 12, 24):
            q = val[:4].double().to(args.device).requires_grad_(True)
            b = posterior(model, q, float(scales[j]), "posterior")
            trace = sum(torch.autograd.grad(b[:,d].sum(), q, retain_graph=d<q.shape[1]-1)[0][:,d]
                        for d in range(q.shape[1]))
            np.testing.assert_allclose(trace.detach().cpu().numpy(), arrays["validation_response"][j,:4], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(b.detach().cpu().numpy(), arrays["validation_posterior"][j,:4], rtol=1e-10, atol=1e-10)
            replays.append(dict(state=state, scale=float(scales[j]),
                max_trace_error=float(abs(trace.detach().cpu().numpy()-arrays["validation_response"][j,:4]).max())))
        noisy_blocks = {key: [] for key in ("posterior", "response", "correction", "full")}
        for j, scale in enumerate(scales):
            noisy_q = noisy_clean+float(scale)*z
            values = readouts(model, noisy_q.to(args.device), scale, args.batch_size)
            for key in noisy_blocks:
                noisy_blocks[key].append(values[key])
            metrics = noisy_metrics(values["posterior"], values["response"], noisy_clean.numpy(), z.numpy(), float(scale))
            metrics["weighted_moment"] = gamma*metrics["moment_mse_div_D"]
            metrics["candidate_combined_objective"] = metrics["original_denoising_loss"]+metrics["weighted_moment"]
            risk_rows.append(dict(state=state, scale=float(scale), query_count=len(noise_ids), **metrics))
            for split in ("validation", "test"):
                row = dict(state=state, split=split, scale=float(scale), query_count=1000)
                for key in ("response", "full"):
                    v = arrays[split+"_"+key][j]
                    row[key+"_mae_to_dimension"] = float(abs(v-DIMENSIONS[dataset]).mean())
                    row[key+"_mean"] = float(v.mean())
                row["correction_mean"] = float(arrays[split+"_correction"][j].mean())
                all_rows.append(row)
        arrays.update({"noisy_"+key: np.stack(value) for key, value in noisy_blocks.items()})
        active_rows, offspan_rows, active_c_rows, offspan_c_rows = [], [], [], []
        for k, j in enumerate(oracle_indices):
            ids = oracle["query_ids"]
            row = dict(state=state, scale=float(scales[j]), query_count=len(ids))
            for key in ("response", "correction", "full"):
                error = arrays["test_"+key][j,ids]-oracle[key][k]
                row[key+"_mae_to_oracle"] = float(abs(error).mean())
                row[key+"_bias_to_oracle"] = float(error.mean())
            error = arrays["test_posterior"][j,ids]-oracle["posterior"][k]
            row["posterior_error_sq_over_lambda_sq"] = float(np.square(error).sum(-1).mean()/scales[j]**2)
            active_r = active_span_trace(model, test[ids].double().to(args.device), scales[j],
                                         active_dimension, args.batch_size)
            offspan_r = arrays["test_response"][j,ids]-active_r
            displacement = arrays["test_posterior"][j,ids]-test[ids].numpy()
            active_c = np.square(displacement[:,:active_dimension]).sum(-1)/scales[j]**2
            offspan_c = np.square(displacement[:,active_dimension:]).sum(-1)/scales[j]**2
            np.testing.assert_allclose(active_c+offspan_c, arrays["test_correction"][j,ids], rtol=1e-12, atol=1e-12)
            row.update(active_span_dimension=active_dimension, active_span_response_mean=float(active_r.mean()),
                active_span_response_mae_to_oracle=float(abs(active_r-oracle["response"][k]).mean()),
                offspan_response_mean=float(offspan_r.mean()), offspan_response_absolute_mean=float(abs(offspan_r).mean()),
                active_span_correction_mean=float(active_c.mean()), offspan_correction_mean=float(offspan_c.mean()))
            active_rows.append(active_r); offspan_rows.append(offspan_r)
            active_c_rows.append(active_c); offspan_c_rows.append(offspan_c)
            oracle_rows.append(row)
        arrays.update(oracle_active_span_response=np.stack(active_rows), oracle_offspan_response=np.stack(offspan_rows),
            oracle_active_span_correction=np.stack(active_c_rows), oracle_offspan_correction=np.stack(offspan_c_rows),
            oracle_query_ids=oracle["query_ids"], oracle_scale_indices=oracle_indices)
        for key in ("response", "full"):
            j, risks = select_scale(arrays["validation_"+key], DIMENSIONS[dataset], scales)
            selections.append(dict(state=state, readout=key, selected_index=j, selected_scale=float(scales[j]),
                at_grid_boundary=j in (0,24), validation_count=1000, test_count=1000,
                validation_mae=float(risks[j]), test_mae=float(abs(arrays["test_"+key][j]-DIMENSIONS[dataset]).mean()),
                all_scale_validation_mae=risks.tolist()))
        np.savez_compressed(args.output/(state+".npz"), scales=scales,
            validation_ids=np.arange(1000), test_ids=np.arange(1000), **arrays)
        print(json.dumps(dict(state=state, completed=True, elapsed_seconds=time.monotonic()-started)), flush=True)
        del model
    assert all(sha(path) == digest for path, digest in hashes.items())
    for filename, rows in (("all_scale.csv", all_rows), ("oracle.csv", oracle_rows), ("noisy_risk.csv", risk_rows)):
        write_csv(args.output/filename, rows)
    (args.output/"selection.json").write_text(json.dumps(selections, indent=2)+"\n")
    training_summary = {}
    tail = history[-int(final["config"]["average_steps"]):]
    for arm in ("control", "moment"):
        training_summary[arm] = {key: float(np.mean([row[arm][key] for row in tail]))
            for key in ("fm", "moment", "weighted_moment", "total")}
    summary = {"all_scale": all_rows, "oracle": oracle_rows, "noisy_risk": risk_rows,
        "selection": selections, "training_tail_means": training_summary,
        "training_tail_scope": "Actual last averaging-window stochastic minibatch telemetry, not the fixed held-out noisy bank"}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    manifest.update(completed=True, sources_unchanged=True, ordinary_autograd_replays=replays,
        original_last_cached_readout_replays=cache_replays,
        elapsed_seconds=time.monotonic()-started,
        outputs={p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!="manifest.json"})
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-replicates", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=2026090717)
    run(parser.parse_args())

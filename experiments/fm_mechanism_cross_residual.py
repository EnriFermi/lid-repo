"""Cross boundary residuals with frozen native FM sensitivity/Adam geometry.

This is a local diagnostic, not a resumed training update. Each recipient
network supplies its own parameter Jacobian and frozen Adam second moment;
the donor supplies only detached canonical posterior errors at identical
observed noisy inputs. All cells share one eta per checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_mechanism_gradient_transfer import adam_metric, inner_product, masked_direction, sample
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, population_posterior
from experiments.fm_mechanism_readout_transfer import readout
from experiments.fm_mechanism_regional_transfer import observed_regions


def residual_gradient(field, inputs, residual, scale, schedule, selected, batch_size=256):
    """Return 2/(N D lambda^2) sum J_i^T e_i with a detached donor e."""
    if inputs.shape != residual.shape or selected.shape != (len(inputs),):
        raise ValueError("Input, residual and mask shapes do not agree")
    if selected.dtype != torch.bool:
        raise ValueError("Source mask must be boolean")
    field.zero_grad(set_to_none=True)
    for begin in range(0, len(inputs), batch_size):
        stop = begin + batch_size
        pred = canonical_posterior(field, inputs[begin:stop], scale, schedule)
        surrogate = (pred * residual[begin:stop].detach()).sum(-1)
        (2 * (surrogate * selected[begin:stop]).sum() / (inputs.numel() * scale**2)).backward()
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in field.parameters()]


def radial_gain_gradient(field, queries, scale, schedule, radial_vectors=None):
    field.zero_grad(set_to_none=True)
    radial_vectors = queries if radial_vectors is None else radial_vectors
    value = canonical_posterior(field, queries, scale, schedule)
    gain = (value * radial_vectors).sum() / radial_vectors.square().sum()
    gradients = torch.autograd.grad(gain, tuple(field.parameters()), allow_unused=True)
    return float(gain.detach()), [torch.zeros_like(p) if g is None else g.detach().clone()
                                 for p, g in zip(field.parameters(), gradients)]


def relative_difference(left, right):
    return math.sqrt(inner_product([a-b for a, b in zip(left, right)],
                                  [a-b for a, b in zip(left, right)])) / max(math.sqrt(inner_product(right, right)), 1e-30)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    fixture_path = args.source.parent / "data.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    if payload["config"].get("input_coordinate", "native") != "native":
        raise ValueError("This diagnostic is specified for native-input fields")
    oracle, _ = make_oracle(payload["config"], fixture, args.device)
    x, z = sample(fixture, 2048, 20260905 + 7101, args.device)
    queries, _ = sample(fixture, 512, 20260905 + 7102, args.device)
    scale = .316227766
    inputs = x + scale*z
    target = population_posterior(inputs, scale, fixture).detach()
    selected = observed_regions(inputs, fixture, scale)["boundary"]
    interior = observed_regions(queries, fixture, scale)["interior"]
    assert int(interior.sum()) >= 8
    schedules = ("rectified_linear", "vp_trigonometric")
    fields, residuals, states = {}, {}, {}
    for schedule in schedules:
        field, _ = load_pilot_checkpoint(args.source, schedule, args.device)
        field = field.double().eval()
        field.pilot_intrinsic = payload["config"]["intrinsic_dim"]
        names, params, metric = adam_metric(field, payload["optimizers"][schedule])
        fields[schedule] = field
        states[schedule] = (names, params, metric, [p.detach().clone() for p in params])
        with torch.no_grad():
            residuals[schedule] = (canonical_posterior(field, inputs, scale, schedule)-target).detach()
    basis = torch.linalg.qr(fixture["tangent_basis"].to(inputs)).Q
    residuals["radial_positive"] = .01 * (inputs @ basis) @ basis.T
    residuals["radial_negative"] = -residuals["radial_positive"]
    residuals["zero"] = torch.zeros_like(inputs)
    residual_rows = []
    radial_coefficients = {}
    source_radial = (inputs[selected] @ basis) @ basis.T
    for name, error in residuals.items():
        e = error[selected]
        tangent = (e @ basis) @ basis.T
        radial_coefficient = float((e*source_radial).sum()/source_radial.square().sum())
        radial_coefficients[name] = radial_coefficient
        residual_rows.append({"donor": name, "count": len(e),
            "boundary_residual_energy": float(e.square().mean()),
            "boundary_tangent_energy": float(tangent.square().mean()),
            "boundary_normal_energy": float((e-tangent).square().mean()),
            "boundary_tangent_radial_coefficient": radial_coefficient,
            "boundary_fitted_tangent_radial_energy": float((radial_coefficient*source_radial).square().mean()),
            "boundary_radial_gain_error": float((e*inputs[selected]).sum()/inputs[selected].square().sum())})
    cosine = float((residuals[schedules[0]][selected]*residuals[schedules[1]][selected]).sum()
                   /(residuals[schedules[0]][selected].norm()*residuals[schedules[1]][selected].norm()))
    gradients, directions, differentials, checks, eta_candidates = {}, {}, {}, [], []
    for schedule in schedules:
        field = fields[schedule]
        names, params, metric, backup = states[schedule]
        _, h = radial_gain_gradient(field, queries[interior], scale, schedule)
        source_radial = (inputs[selected] @ basis) @ basis.T
        _, source_h = radial_gain_gradient(field, inputs[selected], scale, schedule, source_radial)
        for donor, error in residuals.items():
            g = residual_gradient(field, inputs, error, scale, schedule, selected)
            gradients[schedule, donor] = g
            direction = [m*v for m, v in zip(metric, g)]
            directions[schedule, donor] = direction
            if inner_product(direction, direction) > 0:
                eta_candidates.append(args.relative_step * math.sqrt(inner_product(backup, backup)/inner_product(direction, direction)))
            for block in ("all", "conditioning_only", "without_conditioning"):
                adam_direction = masked_direction(names, direction, block)
                euclidean_direction = masked_direction(names, g, block)
                differentials[schedule, donor, block] = {
                    "radial_gain_derivative_adam": -inner_product(h, adam_direction),
                    "radial_gain_derivative_euclidean": -inner_product(h, euclidean_direction),
                    "boundary_radial_gain_derivative_adam": -inner_product(source_h, adam_direction),
                    "boundary_radial_gain_derivative_euclidean": -inner_product(source_h, euclidean_direction),
                    "gradient_norm": math.sqrt(inner_product(g, g)),
                    "masked_direction_norm": math.sqrt(inner_product(adam_direction, adam_direction))}
        pos, neg = gradients[schedule, "radial_positive"], gradients[schedule, "radial_negative"]
        opposite_error = relative_difference(pos, [-v for v in neg])
        assert opposite_error < 1e-12
        assert all(torch.count_nonzero(g) == 0 for g in gradients[schedule, "zero"])
        field.zero_grad(set_to_none=True)
        for begin in range(0, len(inputs), 256):
            pred = canonical_posterior(field, inputs[begin:begin+256], scale, schedule)
            loss = (pred-target[begin:begin+256]).square().mean(-1) / scale**2
            (loss.mul(selected[begin:begin+256]).sum()/len(inputs)).backward()
        own_error = relative_difference(gradients[schedule, schedule], [p.grad for p in params])
        assert own_error < 1e-12
        checks.append({"schedule": schedule, "own_mse_gradient_relative_error": own_error,
                       "opposite_residual_gradient_relative_error": opposite_error, "zero_gradient_exact": True})
    eta = min(eta_candidates)
    attribution_rows = []
    for schedule in schedules:
        for donor in schedules:
            coefficient = radial_coefficients[donor]
            factor = coefficient/.01
            remainder = residuals[donor] - factor*residuals["radial_positive"]
            remainder_gradient = residual_gradient(fields[schedule], inputs, remainder, scale, schedule, selected)
            summed = [a+factor*b for a,b in zip(remainder_gradient, gradients[schedule,"radial_positive"])]
            additivity = relative_difference(summed, gradients[schedule,donor])
            assert additivity < 1e-11
            for block in ("all", "conditioning_only", "without_conditioning"):
                for metric_name in ("adam", "euclidean"):
                    key = "radial_gain_derivative_"+metric_name
                    actual = differentials[schedule,donor,block][key]
                    radial = factor*differentials[schedule,"radial_positive",block][key]
                    attribution_rows.append({"recipient":schedule,"donor":donor,"block":block,
                        "metric":metric_name,"boundary_tangent_radial_coefficient":coefficient,
                        "total_interior_radial_derivative":actual,"fitted_radial_component_derivative":radial,
                        "remainder_derivative":actual-radial,"radial_fraction_signed":radial/actual if actual else 0.,
                        "gradient_additivity_relative_error":additivity})
    rows = []
    for schedule in schedules:
        field = fields[schedule]
        names, params, metric, backup = states[schedule]
        before = readout(field, queries, scale, schedule, fixture, oracle)
        with torch.no_grad():
            before_values = canonical_posterior(field, queries, scale, schedule)
            before_source_values = canonical_posterior(field, inputs[selected], scale, schedule)
        for donor in residuals:
            for block in ("all", "conditioning_only", "without_conditioning"):
                direction = masked_direction(names, directions[schedule, donor], block)
                for fraction in (1., .5):
                    with torch.no_grad():
                        for p, saved, delta in zip(params, backup, direction):
                            p.copy_(saved - fraction*eta*delta)
                        after_values = canonical_posterior(field, queries, scale, schedule)
                        after_source_values = canonical_posterior(field, inputs[selected], scale, schedule)
                    after = readout(field, queries, scale, schedule, fixture, oracle)
                    exact_gain = float(((after_values-before_values)[interior]*queries[interior]).sum()/queries[interior].square().sum())
                    predicted_gain = fraction*eta*differentials[schedule, donor, block]["radial_gain_derivative_adam"]
                    boundary_gain = float(((after_source_values-before_source_values)*source_radial).sum()/source_radial.square().sum())
                    for region in before:
                        rows.append({"recipient": schedule, "donor": donor, "block": block,
                            "fraction": fraction, "eval_region": region, "shared_eta": eta,
                            **differentials[schedule, donor, block],
                            "interior_radial_gain_change": exact_gain,
                            "interior_radial_gain_predicted_change": predicted_gain,
                            "boundary_radial_gain_change": boundary_gain,
                            "interior_over_boundary_radial_gain_change": exact_gain/boundary_gain if boundary_gain else 0.,
                            "interior_radial_linearization_error": abs(exact_gain-predicted_gain)/max(abs(predicted_gain), 1e-30),
                            **{f"before_{k}": before[region][k] for k in before[region] if k != "count"},
                            **{f"change_{k}": after[region][k]-before[region][k] for k in before[region] if k != "count"}})
                    if donor == "zero":
                        assert torch.equal(before_values, after_values) and before == after
                    with torch.no_grad():
                        for p, saved in zip(params, backup):
                            p.copy_(saved)
        assert all(torch.equal(p, saved) for p, saved in zip(params, backup))
        print(json.dumps({"recipient": schedule, "seconds": time.monotonic()-start}), flush=True)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_hash
    write_csv(args.output / "cross_residual.csv", rows)
    write_csv(args.output / "residuals.csv", residual_rows)
    write_csv(args.output / "radial_attribution.csv", attribution_rows)
    manifest = {"source": str(args.source), "source_sha256": source_hash,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scale": scale, "source_count": len(inputs), "boundary_count": int(selected.sum()),
        "query_count": len(queries), "interior_count": int(interior.sum()),
        "source_seed": 20260905+7101, "query_seed": 20260905+7102,
        "shared_eta": eta, "relative_step_ceiling": args.relative_step,
        "donor_residual_cosine_boundary": cosine,
        "source_partition": "observed noisy tangent face-distance <lambda; normalize by ALL2048",
        "intervention": "Boundary donor posterior error, recipient parameter Jacobians AND own frozen inverse-RMS Adam metric; one common eta across all cells",
        "radial_control": "same detached +/-.01 tangent projection of canonical observed inputs for both recipients",
        "derivative_auxiliary": "Euclidean radial-gain derivatives omit Adam metric; no Euclidean full-readout intervention",
        "locality_control": "Boundary self-response uses same968 noisy sources, radial vectors projected onto tangent plane; compare interior gain/boundary gain to remove uniform amplification",
        "residual_decomposition": "Project donor residual onto single shared tangent-radial vector over boundary sources; coefficient times common-radial VJP plus remainder, direct remainder-gradient additivity verified",
        "limits": "Local trained-state sensitivity/residual dissection, not initial distribution shape or accumulated-history cause; no Adam momentum, clipping, decay",
        "checks": checks, "state_restored_bitwise": True, "source_hash_unchanged": True,
        "elapsed_seconds": time.monotonic()-start}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--relative-step", type=float, default=1e-6)
    print(json.dumps(run(parser.parse_args())), flush=True)

"""Diagnostic oracle-risk refit of condition embeddings, with all weights frozen.

One 64-vector per selected scale (or one shared 64-delta) is the only fitted
object. Population oracles are diagnostic supervision, not a deployable method.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_mechanism_crossgeometry_analysis import write_csv
from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_oracles import AffineFixture
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, state_digest


SCALES = (.316227766, 1.)


@contextmanager
def condition_override(field, vector):
    """Override only the computed embedding; retain every original parameter."""
    def replace_output(module, inputs, output):
        if vector.ndim != 1 or vector.shape[0] != output.shape[-1]:
            raise ValueError("Override must be one condition-dimension vector")
        return vector[None, :].expand_as(output)
    handle = field.condition_embedding.register_forward_hook(replace_output)
    try:
        yield
    finally:
        handle.remove()


def draw_clean_noise(config, fixture, count, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    law = config.get("data_law", "gaussian")
    if law == "circle":
        angle = 2*math.pi*torch.rand(count, 1, generator=generator, device=device)
        clean = math.sqrt(2)*torch.cat((angle.cos(), angle.sin()), -1)
    else:
        if law == "gaussian":
            latent = torch.randn(count, config["intrinsic_dim"], generator=generator, device=device)
        else:
            latent = math.sqrt(3)*(2*torch.rand(count, config["intrinsic_dim"], generator=generator, device=device)-1)
        clean = (float(fixture["latent_std"])*latent) @ fixture["tangent_basis"].to(device).T
    noise = torch.randn(count, config["ambient_dim"], generator=generator, device=device)
    return clean.double(), noise.double()


def make_oracle(config, fixture, device):
    law = config.get("data_law", "gaussian")
    if law == "circle":
        def oracle(query, scale):
            return circle_posterior(query, scale, radius=fixture["circle_radius"])
        def tangent_error(error, query):
            normal = query/query.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            tangent = torch.stack((-normal[:, 1], normal[:, 0]), -1)
            return (error*tangent).sum(-1).square()
    else:
        basis = torch.linalg.qr(fixture["tangent_basis"].to(device=device, dtype=torch.float64)).Q
        scale = fixture["latent_half_width"] if law == "uniform" else fixture["latent_std"]
        population = AffineFixture(law, basis, fixture["center"].to(device=device, dtype=torch.float64),
            torch.full((config["intrinsic_dim"],), scale, device=device, dtype=torch.float64))
        oracle = population.posterior
        tangent_error = lambda error, query: (error@basis).square().sum(-1)
    return oracle, tangent_error


def training_target(clean, inputs, scale, oracle, target_mode):
    if target_mode == "clean_X":
        return clean
    if target_mode == "oracle":
        return oracle(inputs, scale).mean.detach()
    raise ValueError("target_mode must be oracle or clean_X")


def fit_vectors(field, schedule, batches, bases, *, shared, max_iter=60):
    vector = torch.nn.Parameter(torch.zeros_like(next(iter(bases.values()))) if shared else next(iter(bases.values())).clone())
    optimizer = torch.optim.LBFGS([vector], lr=1., max_iter=max_iter, max_eval=2*max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
    log = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for scale, (inputs, target) in batches.items():
            embedding = bases[scale]+vector if shared else vector
            with condition_override(field, embedding):
                prediction = canonical_posterior(field, inputs, scale, schedule)
            losses.append((prediction-target).square().mean()/scale**2)
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite fixed-conditioning objective")
        loss.backward()
        log.append({"evaluation":len(log)+1, "loss":float(loss.detach()),
                    "gradient_norm":float(vector.grad.norm())})
        return loss
    started = time.monotonic()
    initial_loss = float(closure().detach())
    optimizer.step(closure)
    final_loss = float(closure().detach())
    assert all(parameter.grad is None for parameter in field.parameters())
    return vector.detach(), {"initial_loss":initial_loss, "final_loss":final_loss,
        "elapsed_seconds":time.monotonic()-started, "iterations":int(optimizer.state[vector]["n_iter"]),
        "function_evaluations":len(log), "log":log,
        "delta_norm":float(vector.norm() if shared else (vector-next(iter(bases.values()))).norm())}


def run(source, output, *, device="cuda:1", fit_count=4096, heldout_count=512, query_count=32,
        max_iter=60, shared=True, target_mode="oracle"):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, output = Path(source), Path(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent/"data.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    law = config.get("data_law", "gaussian")
    assert config.get("output_coordinate", "data") == "data"
    assert config["input_coordinate"] == "native"
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    fit_seed, heldout_seed = config["seed"]+61000, config["seed"]+61001
    clean, noise = draw_clean_noise(config, fixture, fit_count, fit_seed, device)
    held_clean, held_noise = draw_clean_noise(config, fixture, heldout_count, heldout_seed, device)
    oracle, tangent_error = make_oracle(config, fixture, device)
    batches = {scale:(clean+scale*noise, training_target(clean, clean+scale*noise, scale, oracle, target_mode)) for scale in SCALES}
    heldout = {scale:(held_clean+scale*held_noise, oracle(held_clean+scale*held_noise, scale).mean.detach()) for scale in SCALES}
    queries = fixture["clean_queries"][:query_count].to(device=device, dtype=torch.float64)
    if law == "circle":
        queries = fixture["circle_radius"]*queries/queries.norm(dim=-1, keepdim=True)
    torch.save({"fit_clean":clean.cpu(), "fit_noise":noise.cpu(), "heldout_clean":held_clean.cpu(),
                "heldout_noise":held_noise.cpu(), "clean_queries":queries.cpu(),
                "fit_targets":{s:t.cpu() for s,(_,t) in batches.items()},
                "heldout_targets":{s:t.cpu() for s,(_,t) in heldout.items()}}, output/"data.pt")
    manifest = {"source":str(source), "source_sha256":source_hash, "data_law":law,
        "source_config":config, "scales":SCALES, "fit_count":fit_count, "heldout_count":heldout_count,
        "query_count":len(queries), "fit_seed":fit_seed, "heldout_seed":heldout_seed,
        "device":device, "dtype":"float64", "fitted_objects":"one64-vector per scale; optional one shared64-delta",
        "frozen":"All checkpoint parameters, including conditioning projections and original embedding network",
        "optimizer":{"name":"LBFGS","max_iter":max_iter,"line_search":"strong_wolfe","history_size":20},
        "target_mode":target_mode,
        "fit_objective":f"mean-coordinate {target_mode} squared error / lambda^2; shared fit equally weights the two scales",
        "label":"Diagnostic oracle-informed conditioning repair" if target_mode=="oracle" else "Fixed-batch ordinary clean-X denoising-target conditioning repair; oracle used only for evaluation",
        "prediction_before_outcomes":"If learned conditioning mediates residual affine errors,64-vector repair should reduce heldout oracle error without spatial-weight changes; no success assumption for curved support",
        "negative_outcome_policy":"Report heldout worsening without retuning",
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows, saved, arrays = [], {}, {}
    started = time.monotonic()
    for schedule in payload["models"]:
        field,_ = load_pilot_checkpoint(source,schedule,device)
        field = field.double().eval()
        for parameter in field.parameters():
            parameter.requires_grad_(False)
        initial_state = state_digest(field.state_dict())
        with torch.no_grad():
            bases = {scale:field.condition_embedding(queries.new_tensor(math.log(scale)))[0].clone() for scale in SCALES}
        assert all(base.numel()==64 for base in bases.values())
        fits = {"original":bases}
        saved[schedule] = {"original":{s:v.cpu() for s,v in bases.items()},"fits":{}}
        independent = {}
        for scale in SCALES:
            with torch.no_grad():
                original_value = canonical_posterior(field,queries,scale,schedule)
                with condition_override(field,bases[scale]):
                    overridden = canonical_posterior(field,queries,scale,schedule)
                assert torch.allclose(original_value,overridden,atol=1e-11,rtol=1e-11)
            vector,log = fit_vectors(field,schedule,{scale:batches[scale]}, {scale:bases[scale]},shared=False,max_iter=max_iter)
            independent[scale] = vector
            saved[schedule]["fits"][f"independent_{scale:.9g}"] = {"vector":vector.cpu(),**log}
            print(json.dumps({"law":law,"schedule":schedule,"lambda":scale,"mode":"independent",
                              **{k:log[k] for k in ("initial_loss","final_loss","iterations","elapsed_seconds")}}),flush=True)
        fits["independent"] = independent
        if shared:
            delta,log = fit_vectors(field,schedule,batches,bases,shared=True,max_iter=max_iter)
            fits["shared_delta"] = {scale:base+delta for scale,base in bases.items()}
            saved[schedule]["fits"]["shared_delta"] = {"delta":delta.cpu(),**log}
            print(json.dumps({"law":law,"schedule":schedule,"mode":"shared_delta",
                              **{k:log[k] for k in ("initial_loss","final_loss","iterations","elapsed_seconds")}}),flush=True)
        for mode,vectors in fits.items():
            for scale,vector in vectors.items():
                inputs,target = heldout[scale]
                with condition_override(field,vector):
                    with torch.no_grad():
                        prediction = canonical_posterior(field,inputs,scale,schedule)
                        error = prediction-target
                        total_error = error.square().sum(-1)/scale**2
                        tangent = tangent_error(error,inputs)/scale**2
                        clean_pred = canonical_posterior(field,held_clean,scale,schedule)
                        clean_error = clean_pred-oracle(held_clean,scale).mean
                    fn = lambda r: canonical_posterior(field,r,scale,schedule)
                    values,jac = value_and_jacobian(fn,queries)
                response = jac.diagonal(dim1=-2,dim2=-1).sum(-1)
                correction = (values-queries).square().sum(-1)/scale**2
                full = response+correction
                moments = oracle(queries,scale)
                row = {"data_law":law,"schedule":schedule,"mode":mode,"lambda":scale,
                    "heldout_oracle_error_energy_over_lambda2":float(total_error.mean()),
                    "heldout_tangent_oracle_error_energy_over_lambda2":float(tangent.mean()),
                    "heldout_normal_oracle_error_energy_over_lambda2":float((total_error-tangent).mean()),
                    "heldout_clean_oracle_error_energy_over_lambda2":float(clean_error.square().sum(-1).mean()/scale**2),
                    "full_mae_lid":float((full-config["intrinsic_dim"]).abs().mean()),
                    "response_mae_lid":float((response-config["intrinsic_dim"]).abs().mean()),
                    "full_oracle_mae":float((full-moments.full).abs().mean()),
                    "response_oracle_mae":float((response-moments.response).abs().mean()),
                    "full_mean":float(full.mean()),"response_mean":float(response.mean()),
                    "correction_mean":float(correction.mean()),"oracle_correction_mean":float(moments.correction.mean()),
                    "oracle_full_mean":float(moments.full.mean()),
                    "clean_query_oracle_error_energy_over_lambda2":float((values-moments.mean).square().sum(-1).mean()/scale**2)}
                rows.append(row)
                prefix=f"{schedule}__{mode}__{scale:.9g}__"
                for name,value in {"posterior":values,"response":response,"full":full}.items():
                    arrays[prefix+name]=value.detach().cpu().numpy()
                del jac
        assert state_digest(field.state_dict()) == initial_state
        assert not field.condition_embedding._forward_hooks
        saved[schedule]["all_original_parameters_unchanged"] = True
        write_csv(output/"summary.csv",rows)
        torch.save(saved,output/"condition_vectors.pt")
        del field
    lookup={(r["schedule"],r["mode"],r["lambda"]):r for r in rows}
    negative=[{"schedule":r["schedule"],"mode":r["mode"],"lambda":r["lambda"]}
        for r in rows if r["mode"]!="original" and r["heldout_oracle_error_energy_over_lambda2"]>
        lookup[r["schedule"],"original",r["lambda"]]["heldout_oracle_error_energy_over_lambda2"]]
    arrays["queries"]=queries.cpu().numpy()
    np.savez_compressed(output/"query_arrays.npz",**arrays)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    receipt={"status":"complete","elapsed_seconds":time.monotonic()-started,
        "source_unchanged":True,"all_model_parameters_unchanged":True,"heldout_worsening":negative,
        "rows":len(rows),"same_initial_override_verified":True}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--fit-count",type=int,default=4096)
    parser.add_argument("--heldout-count",type=int,default=512)
    parser.add_argument("--query-count",type=int,default=32)
    parser.add_argument("--max-iter",type=int,default=60)
    parser.add_argument("--target-mode",choices=("oracle","clean_X"),default="oracle")
    parser.add_argument("--no-shared",action="store_true")
    args=vars(parser.parse_args());args["shared"]=not args.pop("no_shared")
    run(**args)

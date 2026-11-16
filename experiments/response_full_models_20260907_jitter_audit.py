"""Separate learned antithetic-smoothing error from the operator's own bias."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    oracle=np.load(args.oracle)
    phase=np.load(args.phase_oracle)["query_phase"][:len(oracle["queries"])]
    tangent=tangent_from_phase(phase)
    rows=[]
    for name,folder in [("ve","ve_spaghetti_jitter"),("vp_noise","vp_spaghetti_noise_jitter"),("vp_posterior","vp_spaghetti_posterior_jitter")]:
        learned=np.load(args.root/folder/"arrays.npz")
        tape=np.load(args.root/folder/"noise_tape.npz")
        np.testing.assert_array_equal(learned["queries"],oracle["queries"])
        np.testing.assert_array_equal(learned["query_ids"],oracle["query_ids"])
        np.testing.assert_array_equal(learned["scales"],oracle["scales"])
        np.testing.assert_array_equal(tape["noise"],oracle["noise"])
        baseline_path=args.root/("ve_spaghetti_shared256/e8_spaghetti_pca"if name=="ve"else"vp_spaghetti_span")/("arrays.npz"if name=="ve" else f"{name.removeprefix('vp_')}_average_arrays.npz")
        baseline=np.load(baseline_path)
        for i,scale in enumerate(oracle["scales"]):
            j=int(np.flatnonzero(np.isclose(baseline["scales"],scale,rtol=1e-12,atol=0))[0])
            count=len(oracle["queries"])
            e0=baseline["posterior"][j,:count]-oracle["base_posterior"][i]
            e=learned["posterior"][i]-oracle["posterior"][i]
            # Oracle signs[-,+], learned signs[+,-].
            individual_error=learned["individual_posteriors"][i]-oracle["noisy_posterior"][i,:,:,::-1]
            np.testing.assert_allclose(individual_error.mean((1,2)),e,rtol=1e-10,atol=1e-10)
            mean_individual_energy=(individual_error**2).sum(-1).mean((1,2))/scale**2
            mean_error_energy=(e**2).sum(-1)/scale**2
            centered=(individual_error-e[:,None,None,:])
            variance=(centered**2).sum(-1).mean((1,2))/scale**2
            np.testing.assert_allclose(mean_individual_energy,mean_error_energy+variance,rtol=1e-10,atol=1e-10)
            displacement=oracle["posterior"][i]-oracle["queries"]
            cross=2*(displacement*e).sum(-1)/scale**2
            np.testing.assert_allclose(learned["correction_original"][i],oracle["correction"][i]+cross+mean_error_energy,rtol=1e-9,atol=1e-9)
            baseline_full=baseline["response"][j,:count]+baseline["correction_original"][j,:count]
            tangent_e=(e[:,:20]*tangent).sum(-1)**2/scale**2
            span_e=(e[:,:20]**2).sum(-1)/scale**2
            row={"model":name,"lambda":float(scale),"count":count,
                "baseline_full_d_mae":float(np.abs(baseline_full-1).mean()),
                "jitter_full_d_mae":float(np.abs(learned["full"][i]-1).mean()),
                "original_oracle_full_d_mae":float(np.abs(oracle["base_full"][i]-1).mean()),
                "jitter_operator_oracle_full_d_mae":float(np.abs(oracle["full"][i]-1).mean()),
                "baseline_full_original_oracle_mae":float(np.abs(baseline_full-oracle["base_full"][i]).mean()),
                "jitter_full_original_oracle_mae":float(np.abs(learned["full"][i]-oracle["base_full"][i]).mean()),
                "jitter_full_operator_oracle_mae":float(np.abs(learned["full"][i]-oracle["full"][i]).mean()),
                "baseline_posterior_error_energy":float(((e0**2).sum(-1)/scale**2).mean()),
                "jitter_operator_error_energy":float(mean_error_energy.mean()),
                "individual_noisy_posterior_error_energy":float(mean_individual_energy.mean()),
                "centered_noisy_posterior_error_variation":float(variance.mean()),
                "jitter_operator_error_tangent_energy":float(tangent_e.mean()),
                "jitter_operator_error_curve_normal_energy":float((span_e-tangent_e).mean()),
                "jitter_operator_error_ambient_normal_energy":float(((e[:,20:]**2).sum(-1)/scale**2).mean()),
                "operator_correction_mean":float(oracle["correction"][i].mean()),
                "jitter_correction_mean":float(learned["correction_original"][i].mean()),
                "operator_signed_cross_mean":float(cross.mean())}
            rows.append(row);print(json.dumps(row),flush=True)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),
        "oracle_sha256":sha256(args.oracle),"phase_oracle_sha256":sha256(args.phase_oracle),
        "operator_note":"g*=antithetic smoothing of true b*, not the original Bayes posterior; compare both reference operators",
        "exact_runtime_checks":"Identical tapes/queries/scales; mean error and variance identity; signed correction identity"},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_spaghetti_jitter/arrays.npz"))
    p.add_argument("--phase-oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

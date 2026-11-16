"""Full-batch numerical replay after endpoint tail diagnostics; no model tuning.

The original first128 convergence subset is preserved. This additionally
checks every trajectory of the fixed noise-head states, including large tails.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_transport import integrate
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior


def run(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    rows=[];hashes={str(Path(__file__)):sha256(__file__),
        "experiments/response_full_models_20260907_transport.py":sha256("experiments/response_full_models_20260907_transport.py")}
    for name in ["spaghetti","exp"]:
        source=args.transport/name;fixture=np.load(source/"fixture.npz");a=np.load(source/"endpoints.npz")
        checkpoint=Path("artifacts/response_full_20260907")/f"vp_{name}_paired/final.pt"
        for p in [source/"fixture.npz",source/"endpoints.npz",checkpoint]:hashes[str(p)]=sha256(p)
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        field=load_field(payload,"noise","average",args.device)
        initial=torch.from_numpy(fixture["initial"]).double().to(args.device)
        refined,_,_=integrate(lambda q,s:posterior(field,q,s,"noise"),initial,steps=256,record=False)
        refined=refined.cpu().numpy();difference=np.linalg.norm(refined-a["noise_endpoint"],axis=1)
        scaled=difference/(.01*math.sqrt(initial.shape[1]))
        row={"dataset":name,"count":len(initial),"original_steps":128,"replay_steps":256,
            "mean_difference_over_tube_rms":float(scaled.mean()),"maximum_difference_over_tube_rms":float(scaled.max()),
            "maximum_canonical_l2_difference":float(difference.max()),
            "all_trajectory_gate_passed":float(scaled.mean())<=.02 and float(scaled.max())<=.1}
        rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(args.output/f"{name}.npz",sample_ids=fixture["sample_ids"],endpoint_refined=refined,
            endpoint_refined_raw=refined*float(fixture["normalization_rms"])+fixture["normalization_mean"],
            difference_canonical_l2=difference,difference_over_tube_rms=scaled)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps({"source_hashes":hashes,
        "motivation":"Additional all1024-trajectory numerical validity check after observing noise-head endpoint tails, not changing models/gamma/geometry metrics",
        "selection":"All trajectories of both fixed noise heads, no outlier or query exclusion",
        "primary_results":"Original128-step production endpoints and first128 convergence checks remain immutable",
        "all_sources_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport",type=Path,default=Path("artifacts/response_full_20260907/probability_flow_transport"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())

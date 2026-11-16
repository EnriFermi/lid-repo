"""Describe the removed conditional target residual in the true clean frame.

The frame is anchored at the sampled cleanX, not claimed as the noisy posterior
covariance eigenspace. Training-scale bins are fixed before these summaries.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    before=sha256(args.cache);cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    error=(cache["clean"].double()-cache["oracle_target"].double()).numpy()
    tangent=tangent_from_phase(cache["phase"].numpy());sigma=cache["sigma"].double().numpy()
    radial=np.einsum("bi,bi->b",error[:,:20],tangent)
    weight=(1+sigma*sigma)/(sigma*sigma)/30
    energy={"tangent":radial**2*weight,"curve_normal":((error[:,:20]**2).sum(-1)-radial**2)*weight,
        "ambient_normal":(error[:,20:]**2).sum(-1)*weight}
    total=(error**2).sum(-1)*weight
    np.testing.assert_allclose(sum(energy.values()),total,rtol=1e-11,atol=1e-11)
    bins=[.01,.0316227766,.1,1.,16.];rows=[]
    for low,high in zip(bins[:-1],bins[1:]):
        mask=(sigma>=low)&(sigma<=high if high==16 else sigma<high)
        row={"sigma_low":low,"sigma_high":high,"count":int(mask.sum()),"weighted_label_noise_energy":float(total[mask].mean())}
        for k,v in energy.items():
            row[k+"_weighted_energy"]=float(v[mask].mean())
            row[k+"_fraction"]=float(v[mask].sum()/total[mask].sum())
        rows.append(row)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert sha256(args.cache)==before
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),"cache_sha256":before,
        "frame":"True local tangent at sampled cleanX;19 within-span curve normals;10 absent coordinates",
        "units":"Original EDM weight timescoordinate-mean squared target residual",
        "bins":bins,"scope":"Descriptive removed-target energy, not parameter-gradient covariance or training causality by itself"},indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("artifacts/response_full_20260907/rb_spaghetti_cache/tape.pt"))
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())

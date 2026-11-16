"""Endpoint extrapolation figure from immutable fixed-query arrays."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.response_full_models_20260907_span import sha256


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.7),constrained_layout=True)
    colors={"ve_edm":"#2865ac","vp_noise":"#d47b19","vp_posterior":"#6a3d8f"}
    labels={"ve_edm":"VE / EDM","vp_noise":"VP / noise","vp_posterior":"VP / posterior"}
    sources={}
    for law,style in (("spaghetti","-"),("spiral","--")):
        for family in colors:
            path=args.source/f"{law}_{family}.npz";data=np.load(path);sources[str(path)]=sha256(path)
            x=data["scales"];label=f"{law.capitalize()}: {labels[family]}"
            kw={"color":colors[family],"linestyle":style,"lw":1.6,"label":label}
            axes[0].semilogx(x,data["response"].mean(-1),**kw)
            axes[2].loglog(x,data["correction"].mean(-1),**kw)
            if family!="vp_posterior":
                axes[1].loglog(x,data["canonical_jacobian_minus_identity_frobenius"].max(-1),**kw)
    axes[0].axhline(30,color="black",lw=.8,alpha=.7)
    axes[0].text(1.2e-8,30.8,"ambient D = 30",fontsize=8)
    for ax in axes:
        ax.axvspan(1e-8,.01,color="#c7c7c7",alpha=.22,zorder=-5)
        ax.axvline(.01,color="black",lw=.9,linestyle=":")
        ax.set_xlim(1e-8,1);ax.set_xlabel(r"physical scale $\lambda$")
        ax.grid(alpha=.18,which="major");ax.tick_params(labelsize=8)
        ax.text(.03,.86 if ax is axes[0] else .97,"below training range",transform=ax.transAxes,ha="left",va="top",fontsize=8,color="#555555")
    axes[0].set_ylabel("mean response R")
    axes[1].set_ylabel(r"max over queries $\|J_b-I\|_F$")
    axes[2].set_ylabel("mean correction C (stable algebra)")
    axes[0].set_title("Identity-skip response tends to D",fontsize=10)
    axes[1].set_title("Fixed-model Jacobian tends to identity",fontsize=10)
    axes[2].set_title("Correction need not vanish",fontsize=10)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="outside lower center",ncol=3,fontsize=8)
    fig.savefig(args.output/"endpoint_extrapolation.pdf")
    fig.savefig(args.output/"endpoint_extrapolation.png",dpi=180)
    plt.close(fig)
    (args.output/"manifest.json").write_text(json.dumps({"sources":sources,"source_sha256":sha256(__file__),
        "aggregation":"Mean R/C across fixed64 queries, maximum Jb-I Frobenius across same64",
        "scope":"Shaded region is extrapolation below training support; no geometric oracle comparison"},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/response_full_20260907/endpoint_extrapolation"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

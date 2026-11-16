"""Read-only spectral dissection of preserved posterior Jacobians.

PSD/symmetry violations diagnose local posterior incompatibility, not a
deployable matrix-clipping repair or the historical cause of a value error.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_clean_anchor import write_rows


def error_split(matrix,truth):
    error=matrix-truth;k=error.shape[-1];eye=np.eye(k)
    symmetric=(error+error.swapaxes(-1,-2))/2;skew=(error-error.swapaxes(-1,-2))/2
    mu=np.trace(error,axis1=-2,axis2=-1)/k
    iso=k*mu**2;anisotropic=((symmetric-mu[...,None,None]*eye)**2).sum((-1,-2));curl=(skew**2).sum((-1,-2))
    total=(error**2).sum((-1,-2))
    np.testing.assert_allclose(total,iso+anisotropic+curl,rtol=1e-11,atol=1e-12)
    return {"error_squared":total,"isotropic_gain_error_squared":iso,
        "symmetric_traceless_error_squared":anisotropic,"skew_error_squared":curl}


def spectral(matrix,truth=None):
    symmetric=(matrix+matrix.swapaxes(-1,-2))/2;skew=(matrix-matrix.swapaxes(-1,-2))/2
    eigen=np.linalg.eigvalsh(symmetric);negative=np.minimum(eigen,0.);positive=np.maximum(eigen,0.)
    data={"eigenvalues":eigen,"trace":np.trace(matrix,axis1=-2,axis2=-1),
        "matrix_squared":(matrix**2).sum((-1,-2)),"skew_squared":(skew**2).sum((-1,-2)),
        "negative_mass":-negative.sum(-1),"positive_mass":positive.sum(-1),
        "negative_energy":(negative**2).sum(-1),"positive_energy":(positive**2).sum(-1),
        "negative_count":(eigen < -1e-10).sum(-1),"minimum_eigenvalue":eigen[...,0],
        "distance_squared_to_psd_cone":(negative**2).sum(-1)+(skew**2).sum((-1,-2))}
    np.testing.assert_allclose(data["trace"],data["positive_mass"]-data["negative_mass"],rtol=1e-10,atol=1e-11)
    if truth is not None:
        np.testing.assert_allclose(truth,truth.swapaxes(-1,-2),atol=1e-8,rtol=1e-8)
        assert np.linalg.eigvalsh((truth+truth.swapaxes(-1,-2))/2).min()>-1e-8
        data.update(error_split(matrix,truth))
        data["oracle_trace"]=np.trace(truth,axis1=-2,axis2=-1)
    return data


def complement(tangent):
    projector=np.eye(tangent.shape[-1])[None]-tangent[:,:,None]*tangent[:,None,:]
    _,basis=np.linalg.eigh(projector)
    return basis[:,:,1:]


def restricted(matrix,basis):
    return basis.swapaxes(-1,-2)[None]@matrix@basis[None]


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);rows=[];hashes={str(Path(__file__)):sha256(__file__)};records=[]
    root=args.root
    def load(path,manifest=True):
        path=Path(path);hashes[str(path)]=sha256(path)
        if manifest and path.with_name("manifest.json").exists():hashes[str(path.with_name("manifest.json"))]=sha256(path.with_name("manifest.json"))
        return np.load(path)
    def add(name,law,location,block,matrix,truth,scales,ids,scope):
        result=spectral(matrix,truth);n=matrix.shape[-1]
        for i,s in enumerate(scales):
            values={k+"_mean":float(v[i].mean()) for k,v in result.items() if v.ndim==2}
            values.update({k+"_mean":None for k in ("error_squared","isotropic_gain_error_squared","symmetric_traceless_error_squared","skew_error_squared","oracle_trace") if k not in result})
            rows.append({"state":name,"law":law,"location":location,"block":block,"lambda":float(s),"count":len(ids),"block_dimension":n,
                "oracle_available":truth is not None,"negative_eigenvalue_query_fraction":float((result["negative_count"][i]>0).mean()),
                "minimum_eigenvalue_min":float(result["minimum_eigenvalue"][i].min()),**values})
        np.savez_compressed(args.output/f"{name}_{location}_{block}.npz",scales=scales,query_ids=ids,**result)
        records.append({"state":name,"location":location,"block":block,"scope":scope,"scales":np.asarray(scales).tolist(),"queries":len(ids)})
    # Fresh VP Spaghetti active blocks: all9 saved scales and256 points.
    for head in ("posterior","noise"):
        a=load(root/f"vp_spaghetti_{head}_local/arrays.npz");j=a["learned_jacobian_active"];truth=j-a["jacobian_error_active"]
        basis=complement(a["tangent"])
        for block,b,ref in (("active",j,truth),("curve_normal",restricted(j,basis),restricted(truth,basis))):
            add("vp_"+head,"spaghetti","clean",block,b,ref,a["scales"],a["query_ids"],"Fixed parameter-average model; local normal frame of generating clean phase")
    # Stabilized VE: all9scales; latest deterministic low-LR average on Spaghetti,
    # same-recipe paired-refinement low-LR average on Exp (different ancestry).
    oracle_spag=load(root/"population_oracle_ve_spaghetti256/arrays.npz")
    oracle_exp=load(root/"oracle_eigen_gate/tube_arrays.npz")
    for law,path,oracle,active in (("spaghetti","fixed_objective_lr_spaghetti_audit/low_lr_average.npz",oracle_spag,20),
        ("exp","ve_exp_refinement_audit/low_lr_average.npz",oracle_exp,3)):
        a=load(root/path);j=a["jacobian"];np.testing.assert_array_equal(a["query_ids"],oracle["query_ids"])
        np.testing.assert_allclose(a["scales"],oracle["scales"],rtol=1e-9,atol=0)
        truth=oracle["jacobian_active"]
        basis=complement(a["tangent"]) if law=="spaghetti" else a["normal"][:,:,None]
        blocks=(("active",j[...,:active,:active],truth),
            ("curve_normal",restricted(j[...,:active,:active],basis),restricted(truth,basis)),
            ("ambient_normal",j[...,active:,active:],np.zeros_like(j[...,active:,active:])))
        for block,b,ref in blocks:add("ve_stabilized_"+law,law,"clean",block,b,ref,a["scales"],a["query_ids"],"Known exact finite-scale oracle; stabilized means different explicitly preserved training histories across laws")
    # Spiral: recover canonical J=I+(J-I) from the validated exact-wrapper
    # endpoint arrays, selecting ALL9 in-range scales and no extrapolation.
    for head in ("posterior","noise"):
        a=load(root/f"endpoint_extrapolation/spiral_vp_{head}.npz");mask=~a["out_of_training_range"]
        assert mask.sum()==9
        j=a["canonical_jacobian_minus_identity"][mask]+np.eye(30)
        np.testing.assert_allclose(np.trace(j,axis1=-2,axis2=-1),a["response"][mask],atol=1e-10,rtol=1e-10)
        add("spiral_vp_"+head,"spiral","clean","active",j[...,:2,:2],None,a["scales"][mask],a["query_ids"],"No saved2x2oracle matrix: report PSD/skew violations, not oracle-relative active error")
        add("spiral_vp_"+head,"spiral","clean","ambient_normal",j[...,2:,2:],np.zeros_like(j[...,2:,2:]),a["scales"][mask],a["query_ids"],"Global-null exact posterior derivative is zero; all9 in-range scales only")
    # Matched same64 clean/noisy Spaghetti points: actual conditional oracle at
    # each noisy location, fixed clean-phase frame (not an assumed projector).
    path=Path("../notes/response_full_20260907/spaghetti_noisy_jacobian/arrays.npz");a=load(path,False)
    hashes[str(path.with_name("summary.json"))]=sha256(path.with_name("summary.json"));basis=complement(a["tangent"])
    for name in ("ve","vp_posterior","vp_noise"):
        for location in ("clean","noisy"):
            j=np.stack([a[f"{name}_{i}_{location}"] for i in range(3)])
            truth=np.stack([a[f"oracle_{i}_{location}"] for i in range(3)])
            for block,b,ref in (("active",j[...,:20,:20],truth[...,:20,:20]),
                ("curve_normal",restricted(j[...,:20,:20],basis),restricted(truth[...,:20,:20],basis)),
                ("ambient_normal",j[...,20:,20:],truth[...,20:,20:])):
                add("paired_"+name,"spaghetti",location,block,b,ref,a["scales"],a["query_ids"],"Same64q/Z, same saved3 scales; oracle evaluated at actual input; clean generating-phase frame retained on noisy observations")
    # VP Exp already-saved shell blocks: clean versus typical full Gaussian
    # input. Do not invent a full active3x3 matrix or an all9-scale evaluation.
    for head in ("posterior","noise"):
        a=load(root/f"normal_shell_probe/exp_vp_{head}_arrays.npz")
        for location,span,rho in (("clean",0.,0.),("noisy",1.,1.)):
            si=int(np.flatnonzero(a["span_factors"]==span)[0]);ri=int(np.flatnonzero(a["rhos"]==rho)[0])
            j=a["normal_jacobian"][si,:,ri]
            add("exp_vp_"+head,"exp",location,"ambient_normal",j,np.zeros_like(j),a["scales"],a["query_ids"],"Existing global-normal shell blocks only; clean vs typical full-Gaussian input at4savedscales, not a local normal-geometry comparison")
    write_rows(args.output,"summary",rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"records":records,"completed":True,
        "decomposition":"E=B-B*: ||E||²=k*mu²+||sym(E)-muI||²+||skew(E)||², mu=tr(E)/k",
        "eigenvalues":"Eigenvalues of sym(B), not nonsymmetric eigenvalues; negative-count tolerance1e-10 fixed",
        "scope":"PSD/symmetry are necessary not sufficient for a Gaussian posterior Jacobian. Matrix PSD distance is diagnostic, not an integrable estimator or correction/LID improvement. Skew Jacobian energy is not solenoidal VALUE energy.",
        "no_new_inference_or_training":True,"all_inputs_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())

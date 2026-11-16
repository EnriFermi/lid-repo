"""Exact population counterpart of a fixed antithetic input-jitter readout.

g(x) is the finite average of b_sigma(x +/- sigma*z) at fixed sigma/noise.
It is not silently identified with the original posterior b_sigma(x).
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from experiments.response_full_oracle_20260907_spaghetti import spaghetti_posterior


def main():
    p=argparse.ArgumentParser();p.add_argument('--tape',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tape=dict(np.load(a.tape));queries=tape['queries'].astype(np.float64);noise=tape['noise'].astype(np.float64);scales=tape['scales'];mean=tape['normalization_mean'].astype(np.float64);rms=float(tape['normalization_rms'])
    count,probes,ambient=noise.shape;shape=(len(scales),count);posterior=np.zeros(shape+(ambient,));response=np.zeros(shape);base=np.zeros_like(posterior);base_r=np.zeros(shape)
    b_noisy=np.zeros(shape+(probes,2,ambient));r_noisy=np.zeros(shape+(probes,2));jacs=np.zeros(shape+(20,20));errors=np.zeros(shape+(probes,2));cutoff_errors=np.zeros_like(errors)
    qraw=queries*rms+mean
    for i,scale in enumerate(scales):
        sigma=scale*rms
        for j,q in enumerate(queries):
            central=spaghetti_posterior(qraw[j,:20],sigma)
            central_raw=np.zeros(ambient);central_raw[:20]=central.mean;base[i,j]=(central_raw-mean)/rms;base_r[i,j]=central.response
            for k,z in enumerate(noise[j]):
                for sign_index,sign in enumerate([-1,1]):
                    raw=(q+sign*scale*z)*rms+mean
                    high=spaghetti_posterior(raw[:20],sigma);low=spaghetti_posterior(raw[:20],sigma,order=8)
                    raw_mean=np.zeros(ambient);raw_mean[:20]=high.mean
                    b_noisy[i,j,k,sign_index]=(raw_mean-mean)/rms;r_noisy[i,j,k,sign_index]=high.response
                    jacs[i,j]+=high.posterior_covariance/sigma**2/(2*probes)
                    errors[i,j,k,sign_index]=max(abs(high.response-low.response),abs(high.correction-low.correction))
                    if j%16==0 and k==0:
                        wide=spaghetti_posterior(raw[:20],sigma,cutoff=14)
                        cutoff_errors[i,j,k,sign_index]=max(abs(high.response-wide.response),abs(high.correction-wide.correction))
            posterior[i,j]=b_noisy[i,j].mean(axis=(0,1));response[i,j]=r_noisy[i,j].mean()
        print(float(scale),float(response[i].mean()),flush=True)
    correction=np.square(posterior-queries[None]).sum(axis=-1)/scales[:,None]**2
    base_c=np.square(base-queries[None]).sum(axis=-1)/scales[:,None]**2
    full=response+correction;base_f=base_r+base_c
    np.savez(a.output/'arrays.npz',posterior=posterior,response=response,correction=correction,full=full,jacobian_active=jacs,
             base_posterior=base,base_response=base_r,base_correction=base_c,base_full=base_f,
             noisy_posterior=b_noisy,noisy_response=r_noisy,queries=queries,query_ids=tape['query_ids'],noise=noise,scales=scales,
             normalization_mean=mean,normalization_rms=np.array(rms),order_error=errors,cutoff_error=cutoff_errors)
    report={'tape':str(a.tape),'tape_sha256':hashlib.sha256(a.tape.read_bytes()).hexdigest(),'query_count':count,'antithetic_pairs':probes,
            'definition':'g_sigma(x)=mean_k,sign b_sigma(x+sign*sigma*z_k); same sigma conditioning; derivative holds z fixed; correction centered on original x',
            'schema':'posterior is normalized-data g; base_posterior is original normalized-data b; noisy_posterior[scale,query,pair,sign,30], signs[-1,+1]',
            'max_8_to_16_readout_order_error':float(errors.max()),'max_subsampled_cutoff_error':float(cutoff_errors.max()),
            'summary':[{'scale':float(s),'base_response_mae':float(np.abs(base_r[i]-1).mean()),'base_full_mae':float(np.abs(base_f[i]-1).mean()),
                         'jitter_response_mae':float(np.abs(response[i]-1).mean()),'jitter_full_mae':float(np.abs(full[i]-1).mean()),
                         'base_c_mean':float(base_c[i].mean()),'jitter_c_mean':float(correction[i].mean()),
                         'jitter_posterior_change_energy_over_scale2':float(np.square(posterior[i]-base[i]).sum(axis=1).mean()/s**2)}for i,s in enumerate(scales)]}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

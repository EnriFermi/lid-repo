"""Export exact same-query half-scale posterior and two-scale oracle control."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from experiments.response_full_oracle_20260907 import spiral_posterior,tube_posterior


def halfline_readouts(distance_over_sigma):
    """Uniform local half-line likelihood, including the whole boundary layer.

    The improper flat reference is used only as a local boundary model. Its
    Gaussian-conditioned posterior is proper. Nonnegative distance is inside.
    """
    a=np.asarray(distance_over_sigma,dtype=np.float64)
    if np.any(a<0):raise ValueError('Clean interior/boundary queries require a>=0')
    mills=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)/ndtr(x)
    m=mills(a);mhalf=mills(2*a)
    response=1-a*m-m*m;correction=m*m;first_c=(2*mhalf-m)**2
    return {'response':response,'correction':correction,'full':response+correction,
            'first_c':first_c,'first_full':response+first_c}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=dict(np.load(a.input));q=source['queries_raw'];active=q.shape[1];oracle=spiral_posterior if active==2 else tube_posterior
    raw_sigma=source['raw_sigma'];mean=source['normalization_mean'];rms=float(source['normalization_scale'])
    half=np.zeros_like(source['posterior_raw']);response=np.zeros_like(source['response']);correction=np.zeros_like(response);error=np.zeros_like(response)
    for i,sigma in enumerate(raw_sigma):
        for j,point in enumerate(q):
            low=oracle(point,sigma/2,order=96 if active==2 else 128);high=oracle(point,sigma/2,order=192 if active==2 else 256)
            half[i,j]=high.mean;response[i,j]=high.response;correction[i,j]=high.correction
            error[i,j]=max(abs(high.response-low.response),abs(high.correction-low.correction))
    ambient=np.zeros_like(source['posterior']);ambient[:,:,:active]=half
    half_normalized=(ambient-mean)/rms
    displacement=source['posterior']-source['queries'][None]
    half_displacement=half_normalized-source['queries'][None]
    first_c=np.square(4*half_displacement-displacement).sum(axis=-1)/source['scales'][:,None]**2
    first_full=source['response']+first_c;dimension=1 if active==2 else 2
    np.savez(a.output/'arrays.npz',posterior_half=half_normalized,posterior_half_raw=half,half_response=response,half_correction=correction,
             first_c=first_c,first_full=first_full,query_ids=source['query_ids'],queries=source['queries'],scales=source['scales'],half_scales=source['scales']/2,
             half_raw_sigma=raw_sigma/2,order_error=error,normalization_mean=mean,normalization_scale=np.array(rms))
    report={'source':str(a.input),'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'query_count':len(q),
            'max_order_readout_error':float(error.max()),'schema':{'posterior_half':'[scale,query,ambient] posterior mean in source normalized data coordinates at source.scales/2',
             'posterior_half_raw':'[scale,query,active] posterior mean in raw benchmark coefficient coordinates',
             'first_c':'squared norm of 4*(posterior_half-query)-(posterior_full-query), divided by full normalized scale squared',
             'first_full':'source.response + first_c; source.response is at full scale, not half scale'},
            'summary':[{'scale':float(s),'response_mae':float(np.abs(source['response'][i]-dimension).mean()),'full_mae':float(np.abs(source['full'][i]-dimension).mean()),
                         'first_full_mae':float(np.abs(first_full[i]-dimension).mean()),'first_c_mean':float(first_c[i].mean())}for i,s in enumerate(source['scales'])]}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Fresh continuous Spaghetti tape with converged posterior-mean targets.

This is a nondeployable Rao--Blackwell causal control, not another training
dataset benchmark. Every row is used once. Both arms share clean X, observed
Y, scales and normalization; only X versus E[X|Y] training target differs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_oracle_20260907_spaghetti import spaghetti_posterior


def phase_grid(nodes, device='cpu'):
    theta=torch.arange(nodes,device=device,dtype=torch.float64)*(2*math.pi/nodes)
    harmonics=torch.arange(2,22,device=device,dtype=torch.float64)
    points=torch.sin(theta[:,None]*harmonics)
    return points,points.square().sum(-1)


def grid_posterior(raw_observed, raw_sigma, grid):
    """Probability-normalized uniform-phase quadrature; ambient terms cancel."""
    points,norms=grid
    q=raw_observed.double()
    sigma=torch.as_tensor(raw_sigma,device=q.device,dtype=q.dtype).reshape(-1)
    if q.ndim!=2 or q.shape[1]!=20 or sigma.shape!=(len(q),) or torch.any(sigma<=0):
        raise ValueError('expected Nx20 observations and N positive scales')
    distance=q.square().sum(-1,keepdim=True)+norms[None,:]-2*q@points.T
    weights=torch.softmax(-distance/(2*sigma[:,None].square()),dim=-1)
    return weights@points


def required_nodes(raw_sigma):
    # A resolution heuristic, validated independently; not a global proof.
    lipschitz=math.sqrt(sum(k*k for k in range(2,22)))
    needed=4*math.pi*lipschitz/np.asarray(raw_sigma,dtype=float)
    return np.maximum(512,2.**np.ceil(np.log2(needed))).astype(np.int64)


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    before=sha256(args.source)
    payload=torch.load(args.source,map_location='cpu',weights_only=False)
    # Portable checkpoint owns the coordinate contract.
    mean=payload['normalization']['mean'].float()
    rms=float(payload['normalization']['scale'])
    count=args.steps*args.batch_size
    generator=torch.Generator(device=args.device).manual_seed(args.seed)
    theta=torch.rand(count,device=args.device,dtype=torch.float64,generator=generator)*(2*math.pi)
    raw_x=torch.zeros((count,30),device=args.device,dtype=torch.float64)
    raw_x[:,:20]=torch.sin(theta[:,None]*torch.arange(2,22,device=args.device,dtype=torch.float64))
    clean=((raw_x-mean.double().to(args.device))/rms).float()
    sigma=torch.exp(torch.empty(count,device=args.device,dtype=torch.float32).uniform_(
        math.log(.01),math.log(16.),generator=generator))
    noise=torch.randn(clean.shape,device=args.device,dtype=torch.float32,generator=generator)
    observed=clean+sigma[:,None]*noise
    raw_y=observed.double()*rms+mean.double().to(args.device)
    raw_sigma=sigma.double()*rms
    nodes=required_nodes(raw_sigma.cpu().numpy())
    target=torch.empty_like(clean)
    target[:,20:]=(-mean[20:]/rms).to(args.device)
    t0=time.monotonic(); groups=[]
    with torch.no_grad():
        for size in np.unique(nodes):
            ids=torch.as_tensor(np.flatnonzero(nodes==size),device=args.device)
            grid=phase_grid(int(size),args.device)
            for index in ids.split(args.quadrature_batch):
                value=grid_posterior(raw_y[index,:20],raw_sigma[index],grid)
                target[index,:20]=((value-mean[:20].double().to(args.device))/rms).float()
            groups.append({'nodes':int(size),'count':len(ids),'elapsed_seconds':time.monotonic()-t0})
            print(json.dumps(groups[-1]),flush=True)
    # Ordered scale-spanning checks, selected before looking at target errors.
    order=torch.argsort(sigma).cpu().numpy()
    check_ids=order[np.linspace(0,count-1,min(args.check_count,count),dtype=int)]
    checks=[]
    with torch.no_grad():
        for idx in check_ids:
            q=raw_y[idx:idx+1,:20]
            s=raw_sigma[idx:idx+1]
            n=int(nodes[idx])
            value=grid_posterior(q,s,phase_grid(n,args.device))[0].cpu().numpy()
            double=grid_posterior(q,s,phase_grid(2*n,args.device))[0].cpu().numpy()
            oracle=spaghetti_posterior(q[0].cpu().numpy(),float(s),order=16,cutoff=14.)
            scale=float(s)
            checks.append({'row':int(idx),'sigma':float(sigma[idx]),'nodes':n,
                'doubling_error_l2_over_sigma':float(np.linalg.norm(value-double)/scale),
                'adaptive_error_l2_over_sigma':float(np.linalg.norm(value-oracle.mean)/scale),
                'target_cast_error_l2_over_sigma':float(np.linalg.norm(
                    (target[idx,:20].double().cpu().numpy()*rms+mean[:20].double().numpy())-value)/scale)})
    max_quad=max(max(v['doubling_error_l2_over_sigma'],v['adaptive_error_l2_over_sigma']) for v in checks)
    if max_quad>2e-6: raise RuntimeError(f'Independent oracle target gate failed: {max_quad}')
    assert torch.isfinite(target).all() and sha256(args.source)==before
    tape={'dataset':'e8_spaghetti_pca','clean':clean.cpu(),'observed':observed.cpu(),
          'sigma':sigma.cpu(),'oracle_target':target.cpu(),'phase':theta.cpu(),
          'normalization_mean':mean,'normalization_scale':rms,
          'steps':args.steps,'batch_size':args.batch_size,'seed':args.seed,
          'source_checkpoint_sha256':before,'fresh_continuous_population':True,
          'draw_contract':'one pass, continuous uniform phase; float32 X/Y/sigma; Gaussian observation noise'}
    torch.save(tape,args.output/'tape.pt')
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256':sha256(__file__),
        'adaptive_oracle_sha256':sha256(Path(__file__).with_name('response_full_oracle_20260907_spaghetti.py')),
        'checkpoint_sha256':before,'tape_sha256':sha256(args.output/'tape.pt'),
        'source_checkpoint_unchanged':True,'rows':count,'steps':args.steps,'batch_size':args.batch_size,
        'seed':args.seed,'sigma_range':[.01,16.],'normalization_rms':rms,
        'max_independent_posterior_error_l2_over_sigma':max_quad,
        'max_float32_target_cast_error_l2_over_sigma':max(v['target_cast_error_l2_over_sigma'] for v in checks),
        'scope':'same expected UNCLIPPED parameter gradient under continuous population; clipping and Adam updates are nonlinear',
        'target':'converged continuous uniform-phase conditional mean at the actual cached noisy input',
        'groups':groups,'elapsed_seconds':time.monotonic()-t0},indent=2)+'\n')
    print(f'Complete: {count} fresh rows; target-check maximum {max_quad:.3g}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--quadrature-batch',type=int,default=256)
    parser.add_argument('--check-count',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20260909)
    parser.add_argument('--device',default='cuda:1')
    run(parser.parse_args())

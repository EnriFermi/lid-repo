"""Continuous Spaghetti oracle counterpart of the published noisy-score SVD.

The tape, scales, K=4D, uncentered vectors, and largest absolute adjacent
singular-value drop are fixed by the paired learned-model control.  Tangent
information is used for retrospective evaluation only, never by the estimator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.response_full_oracle_20260907_spaghetti import spaghetti_posterior


def spectral_readout(scores, tangent):
    scores = np.asarray(scores, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < scores.shape[1]:
        raise ValueError('Require at least D vectors, with shape K,D')
    if tangent.shape != (scores.shape[1],):
        raise ValueError('Tangent must have D components')
    if not np.isclose(np.linalg.norm(tangent), 1.):
        raise ValueError('Evaluation tangent must be unit length')
    vectors, singular, _ = np.linalg.svd(scores.T, full_matrices=False)
    gaps = singular[:-1]-singular[1:]
    normal_rank = int(np.argmax(gaps))+1
    dimension = scores.shape[1]-normal_rank
    return {
        'singular_values': singular,
        'dimension': dimension,
        'normal_rank': normal_rank,
        'gap_index': normal_rank-1,
        'gap': float(gaps[normal_rank-1]),
        'gap_over_largest': float(gaps[normal_rank-1]/singular[0]),
        'smallest_vector_tangent_cosine2': float((vectors[:, -1]@tangent)**2),
        'estimated_tangent_capture': float(np.sum((vectors[:, normal_rank:].T@tangent)**2)),
    }


def oracle_query_row(arguments):
    index, query, noise, scale, mean, rms = arguments
    sigma = scale*rms
    observations = query[None, :]+scale*noise
    raw = observations*rms+mean
    posterior = np.empty_like(observations)
    nodes = np.zeros(len(noise), dtype=np.int64)
    score_order_error = 0.
    score_cutoff_error = 0.
    for k, observation in enumerate(raw):
        result = spaghetti_posterior(observation[:20], sigma)
        raw_mean = np.zeros(len(query))
        raw_mean[:20] = result.mean
        posterior[k] = (raw_mean-mean)/rms
        nodes[k] = result.metadata['quadrature_nodes']
        if k == 0:
            low = spaghetti_posterior(observation[:20], sigma, order=8)
            # Dimensionless scale*score error, comparable across scales.
            score_order_error = float(np.linalg.norm(result.mean-low.mean)/sigma)
            if index % 8 == 0:
                wide = spaghetti_posterior(observation[:20], sigma, cutoff=14.)
                score_cutoff_error = float(np.linalg.norm(result.mean-wide.mean)/sigma)
    return posterior, nodes, score_order_error, score_cutoff_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    tape_hash = hashlib.sha256(args.tape.read_bytes()).hexdigest()
    tape = dict(np.load(args.tape))
    queries = tape['queries'].astype(np.float64)
    noise = tape['noise'].astype(np.float64)
    scales = tape['scales'].astype(np.float64)
    mean = tape['normalization_mean'].astype(np.float64)
    rms = float(tape['normalization_rms'])
    phase = tape['query_phase'].astype(np.float64)
    count, probes, ambient = noise.shape
    if queries.shape != (count, ambient) or probes != 4*ambient:
        raise ValueError('Paired protocol requires queries N,D and noise N,4D,D')
    tangent = np.zeros_like(queries)
    frequencies = np.arange(2, 22)
    tangent[:, :20] = np.cos(phase[:, None]*frequencies[None, :])*frequencies
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    np.testing.assert_allclose(tangent, tape['tangent'], atol=1e-12, rtol=1e-12)
    args.output.mkdir(parents=True, exist_ok=False)
    shape = (len(scales), count)
    posterior = np.zeros(shape+(probes, ambient))
    scores = np.zeros_like(posterior)
    singular = np.zeros(shape+(ambient,))
    dimensions = np.zeros(shape, dtype=np.int64)
    cosine = np.zeros(shape)
    capture = np.zeros(shape)
    gaps = np.zeros(shape)
    gap_relative = np.zeros(shape)
    order_error = np.zeros(shape)
    cutoff_error = np.zeros(shape)
    nodes = np.zeros(shape+(probes,), dtype=np.int64)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, scale in enumerate(scales):
            jobs = [(j, query, noise[j], scale, mean, rms) for j, query in enumerate(queries)]
            rows = pool.map(oracle_query_row, jobs)
            for j, (post, n, order, cutoff) in enumerate(rows):
                posterior[i, j] = post
                nodes[i, j] = n
                order_error[i, j] = order
                cutoff_error[i, j] = cutoff
                observed = queries[j][None, :]+scale*noise[j]
                scores[i, j] = (post-observed)/scale**2
                spectral = spectral_readout(scores[i, j], tangent[j])
                singular[i, j] = spectral['singular_values']
                dimensions[i, j] = spectral['dimension']
                cosine[i, j] = spectral['smallest_vector_tangent_cosine2']
                capture[i, j] = spectral['estimated_tangent_capture']
                gaps[i, j] = spectral['gap']
                gap_relative[i, j] = spectral['gap_over_largest']
            print(json.dumps({'scale': float(scale), 'dimension_mae': float(np.abs(dimensions[i]-1).mean()),
                              'tangent_cosine2_mean': float(cosine[i].mean()),
                              'elapsed_seconds': time.perf_counter()-started}), flush=True)
    if hashlib.sha256(args.tape.read_bytes()).hexdigest() != tape_hash:
        raise RuntimeError('Input tape changed during evaluation')
    np.savez(args.output/'arrays.npz', posterior=posterior, scores=scores,
             singular_values=singular, dimension=dimensions, tangent_cosine2=cosine,
             estimated_tangent_capture=capture, largest_absolute_gap=gaps,
             gap_over_largest_singular=gap_relative, tangent=tangent, queries=queries,
             query_ids=tape['query_ids'], phase=phase, noise=noise, scales=scales,
             normalization_mean=mean, normalization_rms=np.array(rms),
             order_error_scale_times_score=order_error, cutoff_error_scale_times_score=cutoff_error,
             quadrature_nodes=nodes)
    summary = []
    for i, scale in enumerate(scales):
        unique, counts = np.unique(dimensions[i], return_counts=True)
        summary.append({'scale': float(scale), 'dimension_mae': float(np.abs(dimensions[i]-1).mean()),
                        'dimension_counts': {str(k): int(v) for k, v in zip(unique, counts)},
                        'tangent_cosine2_mean': float(cosine[i].mean()),
                        'tangent_cosine2_min': float(cosine[i].min()),
                        'largest_gap_relative_mean': float(gap_relative[i].mean())})
    report = {'tape': str(args.tape), 'tape_sha256': tape_hash, 'query_count': count,
              'probe_count': probes, 'ambient_dimension': ambient, 'true_dimension': 1,
              'protocol': 'No centering; columns (b*(x+lambda*z)-x-lambda*z)/lambda²; '
                          'largest absolute adjacent singular-value drop; d=D-(gap_index+1).',
              'scope': 'Continuous exact-law oracle; paired fixed tape. True tangent used only for evaluation.',
              'max_order_error_scale_times_score': float(order_error.max()),
              'max_subsampled_cutoff_error_scale_times_score': float(cutoff_error.max()),
              'elapsed_seconds': time.perf_counter()-started, 'summary': summary}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Continuous closest-point diagnostics for the actual Exp and Spaghetti laws.

Inputs are raw benchmark coordinates, before model mean/RMS normalization.
No neural model or LID label is used.  Distances include all extra ambient
coordinates.  Exp has a complete stationary-point enumeration. Spaghetti
uses competitive coarse brackets, bounded minimization, stationary refinement,
and a doubled-grid convergence check, not an asserted symbolic global proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar


HARMONICS = np.arange(2., 22.)


def _points(points, active):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < active or not np.isfinite(points).all():
        raise ValueError('Require finite raw points with shape N,D and enough active coordinates')
    return points


def _padded(rows):
    width = max(map(len, rows), default=0)
    result = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        result[i, :len(row)] = row
    return result


def _candidate_summary(latents, distances, *, period=None, tolerance=1e-10):
    order = np.argsort(distances)
    unique_latents, unique_distances = [], []
    for index in order:
        value = float(latents[index])
        if period is not None:
            value %= period
        difference = np.abs(value-np.asarray(unique_latents))
        if period is not None:
            difference = np.minimum(difference, period-difference)
        if len(difference) and difference.min() < 1e-8:
            continue
        unique_latents.append(value)
        unique_distances.append(float(distances[index]))
    gap = unique_distances[1]-unique_distances[0] if len(unique_distances)>1 else np.inf
    ambiguous = gap <= tolerance*(1+unique_distances[0])
    return np.asarray(unique_latents), np.asarray(unique_distances), ambiguous, gap


def closest_tube(points):
    """Global Exp projection: radius=3exp(-u),u in[0,8], uniform angle.

    Minimize (u-4-x0)^2+(3exp(-u)-rho)^2. Half its derivative is
    g=u-4-x0+rho*r-r²; g'=1-rho*r+2r² has at most two zeros.
    Partition at those zeros and bracket every root of g on each monotone
    interval. Endpoints plus all stationary points contain the global minimum.
    """
    points = _points(points, 3)
    nearest = np.zeros_like(points)
    latent_u, theta, ambiguous, gap = [], [], [], []
    candidates, candidate_distance = [], []
    angular_ambiguity = []
    for query in points:
        rho = float(np.linalg.norm(query[1:3]))
        angle = float(np.arctan2(query[1], query[2]) % (2*np.pi)) if rho else 0.
        breaks = [0., 8.]
        if rho >= np.sqrt(8.):
            discriminant = np.sqrt(rho*rho-8.)
            radii = [(rho+discriminant)/4., 2./(rho+discriminant)]
            breaks += [-np.log(r/3.) for r in radii if 3*np.exp(-8.) < r < 3.]
        breaks = np.unique(breaks)
        def objective(u):
            r = 3*np.exp(-u)
            return (u-4-query[0])**2+(r-rho)**2
        def derivative(u):
            r = 3*np.exp(-u)
            return u-4-query[0]+rho*r-r*r
        values = list(breaks)
        for left, right in zip(breaks[:-1], breaks[1:]):
            if derivative(left)*derivative(right) < 0:
                values.append(brentq(derivative, left, right, xtol=1e-14, rtol=1e-14))
        values, distances, tied, distance_gap = _candidate_summary(values, [objective(u) for u in values])
        u = values[0]
        latent_u.append(u); theta.append(angle); ambiguous.append(tied); gap.append(distance_gap)
        candidates.append(values); candidate_distance.append(distances)
        angular_ambiguity.append(rho == 0.)
    latent_u = np.asarray(latent_u); theta = np.asarray(theta)
    radius = 3*np.exp(-latent_u)
    nearest[:, 0] = latent_u-4.
    nearest[:, 1] = radius*np.sin(theta)
    nearest[:, 2] = radius*np.cos(theta)
    active = np.square(nearest[:, :3]-points[:, :3]).sum(axis=1)
    ambient = np.square(points[:, 3:]).sum(axis=1)
    return {'nearest_raw': nearest, 'distance_squared': active+ambient,
            'active_distance_squared': active, 'ambient_distance_squared': ambient,
            'u': latent_u, 'theta': theta, 'ambiguous_projection': np.asarray(ambiguous),
            'ambiguous_angle': np.asarray(angular_ambiguity), 'second_candidate_distance_gap': np.asarray(gap),
            'candidate_u': _padded(candidates), 'candidate_active_distance_squared': _padded(candidate_distance),
            'candidate_count': np.asarray(list(map(len, candidates)), dtype=np.int64)}


def _spaghetti_grid(points, grid_size):
    step = 2*np.pi/grid_size
    phase = step*np.arange(grid_size)
    curve = np.sin(phase[:, None]*HARMONICS)
    # Vectorized coarse distance matrix; local objectives below recompute
    # differences directly to avoid small-distance cancellation.
    coarse = np.maximum(0., np.square(points).sum(1)[:, None]
                        +np.square(curve).sum(1)[None, :]-2*points@curve.T)
    lipschitz = np.linalg.norm(HARMONICS)
    latent_rows, distance_rows, ambiguous, gaps = [], [], [], []
    for query, row in zip(points, coarse):
        indices = np.flatnonzero((row <= np.roll(row, 1)) & (row <= np.roll(row, -1)))
        # If the doubled-cell neighborhood cannot improve the best sampled
        # point even under the global derivative bound, discard that bracket.
        lower = np.maximum(0., np.sqrt(row[indices])-lipschitz*step)**2
        indices = indices[lower <= row.min()+1e-12]
        if not len(indices):
            indices = np.array([np.argmin(row)])
        def objective(t):
            return float(np.square(np.sin(HARMONICS*t)-query).sum())
        def derivative(t):
            return float(((np.sin(HARMONICS*t)-query)*HARMONICS*np.cos(HARMONICS*t)).sum())
        def second_derivative(t):
            sine, cosine = np.sin(HARMONICS*t), np.cos(HARMONICS*t)
            return float((HARMONICS**2*(cosine**2-(sine-query)*sine)).sum())
        candidates = []
        for index in indices:
            middle = phase[index]
            left, right = middle-step, middle+step
            optimum = minimize_scalar(objective, bounds=(left, right), method='bounded',
                                      options={'xatol': 1e-14})
            candidate = float(optimum.x)
            # Bounded Brent uses a sqrt(machine-eps)*|phase| stopping term;
            # polish the stationary equation to avoid that coordinate floor.
            for _ in range(5):
                curvature = second_derivative(candidate)
                if curvature <= 0.:
                    break
                refined = candidate-derivative(candidate)/curvature
                if not left <= refined <= right:
                    break
                candidate = refined
            candidates.append(candidate)
            if derivative(left) <= 0. <= derivative(right):
                candidates.append(brentq(derivative, left, right, xtol=1e-14, rtol=1e-14))
        values, distances, tied, gap = _candidate_summary(candidates, [objective(t) for t in candidates], period=2*np.pi)
        latent_rows.append(values); distance_rows.append(distances); ambiguous.append(tied); gaps.append(gap)
    return latent_rows, distance_rows, np.asarray(ambiguous), np.asarray(gaps)


def closest_spaghetti(points, *, grid_size=1024, refinement_grid_size=2048, tolerance=1e-9):
    """Numerically converged closest points on the full uniform-phase curve.

    The support is continuous, not the coarse phase grid. Returned candidates
    are the competitive grid-bracket solutions, not all algebraic stationary
    points. Opposite branches meet at the origin, so phase can be nonunique.
    A failed grid-convergence check raises instead of silently certifying it.
    """
    points = _points(points, 20)
    if grid_size < 128 or refinement_grid_size <= grid_size:
        raise ValueError('Require a meaningful grid and a finer convergence grid')
    first = _spaghetti_grid(points[:, :20], grid_size)
    refined = _spaghetti_grid(points[:, :20], refinement_grid_size)
    low = np.array([row[0] for row in first[1]])
    high = np.array([row[0] for row in refined[1]])
    error = np.abs(low-high)
    if np.any(error > tolerance*(1+high)):
        raise RuntimeError(f'Closest-point grid check failed: max squared-distance difference {error.max()}')
    phases = np.array([row[0] for row in refined[0]])
    nearest = np.zeros_like(points)
    nearest[:, :20] = np.sin(phases[:, None]*HARMONICS)
    active = np.square(nearest[:, :20]-points[:, :20]).sum(1)
    ambient = np.square(points[:, 20:]).sum(1)
    return {'nearest_raw': nearest, 'distance_squared': active+ambient,
            'active_distance_squared': active, 'ambient_distance_squared': ambient,
            'phase': phases, 'ambiguous_projection': refined[2],
            'second_candidate_distance_gap': refined[3],
            'candidate_phase': _padded(refined[0]),
            'candidate_active_distance_squared': _padded(refined[1]),
            'candidate_count': np.asarray(list(map(len, refined[0])), dtype=np.int64),
            'grid_refinement_squared_distance_error': error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    hashes = {}
    start = time.perf_counter()
    for name, helper in [('spaghetti', closest_spaghetti), ('exp', closest_tube)]:
        source = args.transport_root/name
        status = json.loads((source/'status.json').read_text())
        if status['status'] != 'complete' or not status['all_numerical_gates_passed']:
            raise ValueError('Require completed transport and passing numerical gates')
        for file in ['fixture.npz', 'endpoints.npz', 'manifest.json', 'status.json']:
            path = source/file
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixture = dict(np.load(source/'fixture.npz'))
        endpoints = dict(np.load(source/'endpoints.npz'))
        raw = {'reference': fixture['reference_raw']}
        raw.update({key.removesuffix('_endpoint_raw'): value for key, value in endpoints.items()
                    if key.endswith('_endpoint_raw')})
        target = args.output/name; target.mkdir()
        for variant, points in raw.items():
            result = helper(points)
            np.savez_compressed(target/(variant+'.npz'), raw_points=points,
                                sample_ids=fixture['sample_ids'], normalization_rms=fixture['normalization_rms'], **result)
            row = {'dataset': name, 'variant': variant, 'count': len(points),
                   'distance_rms_canonical': float(np.sqrt(result['distance_squared'].mean())/fixture['normalization_rms']),
                   'active_distance_rms_canonical': float(np.sqrt(result['active_distance_squared'].mean())/fixture['normalization_rms']),
                   'ambient_distance_rms_canonical': float(np.sqrt(result['ambient_distance_squared'].mean())/fixture['normalization_rms']),
                   'ambiguous_projection_count': int(result['ambiguous_projection'].sum()),
                   'max_grid_squared_distance_difference': float(result.get('grid_refinement_squared_distance_error', np.array([0.])).max())}
            summary.append(row)
            print(json.dumps(row), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
        raise RuntimeError('Transport source changed during geometric audit')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({
        'source_hashes': hashes, 'geometry_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Continuous support distance and projected latent coordinates; not full generative distribution certification. '
                 'Reference is the true p_.01 marginal, not zero-distance clean support.',
        'spaghetti': 'Continuous local minima from1024 and2048 global grids, stationary-polished, checked for equal minimum distance; no symbolic global guarantee.',
        'exp': 'Complete stationary-root enumeration using monotone intervals of the scalar derivative; endpoints included.',
        'all_inputs_unchanged': True, 'elapsed_seconds': time.perf_counter()-start}, indent=2)+'\n')


if __name__ == '__main__':
    main()

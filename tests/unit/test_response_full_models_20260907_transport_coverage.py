import numpy as np
from experiments.response_full_models_20260907_transport_coverage import circular_summary,tube_coverage


def test_fixed_circle_coverage_flags_collapse_but_reference_is_zero_distance():
    reference=np.linspace(0,2*np.pi,1024,endpoint=False)
    same,_=circular_summary(reference,reference)
    collapsed,_=circular_summary(np.zeros(1024),reference)
    assert same["histogram32_tv"]==0 and same["histogram32_occupied"]==32
    assert collapsed["histogram32_occupied"]==1 and collapsed["histogram32_tv"]>.95
    assert collapsed["largest_moment1to8_magnitude"]==1


def test_tube_joint_can_detect_dependence_with_unchanged_marginals():
    u=np.repeat(np.arange(8)+.5,8);theta=np.tile((np.arange(8)+.5)*2*np.pi/8,8)
    altered=np.repeat((np.arange(8)+.5)*2*np.pi/8,8)
    summary,_=tube_coverage(u,altered,u,theta)
    assert summary["u_histogram32_tv"]==0 and summary["theta_histogram32_tv"]==0
    assert summary["u_theta_joint8x8_tv"]>.8

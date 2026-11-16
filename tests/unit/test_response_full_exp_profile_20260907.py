import numpy as np
from experiments.response_full_exp_profile_20260907 import tube_geometry, decompose, summarize, status


def test_exact_frame_matches_surface_derivatives_and_boundaries():
    u=np.array([0.,1.,4.,8.]);t=np.array([.1,.7,1.2,2.])
    r=3*np.exp(-u);q=np.column_stack((u-4,r*np.sin(t),r*np.cos(t)))
    g=tube_geometry(q);f=g['frame']
    np.testing.assert_allclose(f.transpose(0,2,1)@f,np.broadcast_to(np.eye(3),f.shape),atol=1e-14)
    np.testing.assert_allclose(f[:,:,0]*np.sqrt(1+r*r)[:,None],
        np.column_stack((np.ones(4),-r*np.sin(t),-r*np.cos(t))),atol=1e-14)
    np.testing.assert_allclose(g['boundary_distance'][[0,-1]],0,atol=1e-14)


def test_signed_error_split_and_negative_cross_in_every_frame_component():
    rng=np.random.default_rng(1);q=np.zeros((5,30));scales=np.array([.1,.4])
    frame=np.broadcast_to(np.eye(3),(5,3,3));d=rng.normal(size=(2,5,30))*.1
    star=q[None]+d;model=q[None]+.4*d
    out=decompose(q,model,star,np.ones((2,5)),np.ones((2,5)),scales,frame)
    np.testing.assert_allclose(out['correction'],.16*out['oracle_correction'])
    assert np.all(out['cross']<0)
    np.testing.assert_allclose(out['cross'],sum(out['cross_'+p] for p in ('meridional','circumferential','surface_normal','ambient')))


def test_help_harm_reversal_can_come_from_wrong_response_not_correction():
    q=np.zeros((1,3));scales=np.ones(1);frame=np.eye(3)[None]
    star=np.array([[[np.sqrt(.6),0,0]]])
    a=decompose(q,star,star,np.array([[2.1]]),np.array([[1.4]]),scales,frame)
    assert a['oracle_status'].item()==1 and a['status'].item()==-1
    assert a['substitute_exact_c_status'].item()==-1


def test_binning_partition_and_pointwise_joint_counts():
    u=np.array([.1,2.,4.,6.,8.]);r=3*np.exp(-u)
    q=np.column_stack((u-4,r,np.zeros(5)));g=tube_geometry(q)
    q30=np.pad(q,((0,0),(0,27)));rhat=np.full((1,5),1.4)
    star=q30[None].copy();star[:,:,0]+=np.sqrt(.6)
    a=decompose(q30,star,star,rhat,rhat,np.ones(1),g['frame'])
    rows=summarize(a,g,np.array([.1]),'test')
    bins=[row for row in rows if row['group'].startswith('u_')]
    assert sum(row['count'] for row in bins)==5
    assert [row['count'] for row in bins]==[1,1,1,2]
    for row in rows:
        if row['count']:
            assert sum(v for k,v in row.items() if k.startswith('oracle_') and '_model_' in k and k.endswith('_count'))==row['count']

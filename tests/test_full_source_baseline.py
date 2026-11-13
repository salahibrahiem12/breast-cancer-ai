import numpy as np
from scipy.optimize._numdiff import approx_derivative
from ml.full_source_baseline import objective, sample_weights


def test_weighted_logistic_gradient_matches_finite_differences():
    z=np.array([[1.,2.],[-1.,.3],[.2,-.7],[2.,-.5]])
    y=np.array([0,1,1,0]);w=np.array([.5,1.5,.3,.7]);theta=np.array([.2,-.3,.1])
    value,gradient=objective(theta,z,y,w)
    numerical=approx_derivative(lambda t: objective(t,z,y,w)[0],theta).ravel()
    np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-8)
    assert np.isfinite(value)


def test_source_diagnosis_group_mass_survives_duplicate_images():
    source=np.array(['a','a','a','a','b','b','b'])
    label=np.array([0,0,0,1,0,1,1]);group=np.array(['g1','g1','g2','g3','g4','g5','g5'])
    w=sample_weights(label,source,group,True)
    for s in ['a','b']:
        for c in [0,1]:
            assert np.isclose(w[(source==s)&(label==c)].sum(),5/4)
    assert np.isclose(w[0]+w[1],w[2])
    assert np.isclose(w.sum(),5)


def test_calibration_weighting_preserves_natural_group_prevalence():
    source=np.array(['a','a','a','a','b','b','b'])
    label=np.array([0,0,0,1,0,1,1]);group=np.array(['g1','g1','g2','g3','g4','g5','g5'])
    w=sample_weights(label,source,group,False)
    assert np.isclose(w[source=='a'].sum(),2.5)
    assert np.isclose(w[source=='b'].sum(),2.5)
    assert np.isclose(w[label==1].sum()/w.sum(),(.3333333333333333+.5)/2)

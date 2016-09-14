
from delight.utils import *
from astropy.cosmology import FlatLambdaCDM


def test_approx_DL():
    for z in np.linspace(0.01, 4, num=10):
        z = 2.
        v1 = approx_DL()(z)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None)
        v2 = cosmo.luminosity_distance(z).value
        assert abs(v1/v2 - 1) < 0.01


def test_random_X():
    size = 10
    X = random_X_bzl(size, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 3)

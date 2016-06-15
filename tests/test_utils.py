
from delight.utils import random_X_bzlt


def test_random_X():
    size = 10
    X = random_X_bzlt(size, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 4)

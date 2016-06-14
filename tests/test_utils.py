
from delight.utils import random_X_bzlt


def test_random_X():
    size = 10
    X = random_X_bzlt(size, numTypes=8, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 4)

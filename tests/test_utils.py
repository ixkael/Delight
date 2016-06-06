
from delight.utils import random_X_bztl


def test_random_X():
    size = 10
    X = random_X_bztl(size, numTypes=8, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 4)

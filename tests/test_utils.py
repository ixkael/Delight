
#from .context import delight
from delight.utils import *

def test_random_X():
    size = 10
    X = random_X_tbz(size, numTypes=8, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 3)
    X = random_X_zl(size)
    assert X.shape == (size, 2)


#from .context import delight
from delight.utils import *

def test_random_X():
    X = random_X(10, numTypes=8, numBands=5, redshiftMax=3.0)
    assert X.shape == (10, 3)

from importlib import metadata
from bim.Bimbalance import bSFS, bTree
from bim.utils import Colless, Neutrality_Tests

tstats = ['btree', 'Colless', 'Omega']
sstats = ['bsfs', 'TajD', 'FayH', 'ZngE', 'FerL', 'FulD']


__version__ = metadata.metadata("bim")["Version"]
__author__ = metadata.metadata("bim")["Author"]
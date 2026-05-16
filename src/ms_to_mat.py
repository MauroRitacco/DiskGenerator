# Import required tools/tasks
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys

# Workaround for casatasks MatplotlibDeprecationWarning with newer matplotlib versions
import matplotlib.cbook
if not hasattr(matplotlib.cbook, "MatplotlibDeprecationWarning"):
    matplotlib.cbook.MatplotlibDeprecationWarning = DeprecationWarning

from casatasks import tclean, ft, imhead, listobs, exportfits, flagdata, bandpass, applycal
from casatasks.private import simutil

import os
import pylab as pl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.io import loadmat


# Instantiate all the required tools
sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
cl = componentlist()
mysu = simutil.simutil()

# Check what columns are in the ms file
def inspectMsColumns(ms):
    tb.open(ms)
    colnames = tb.colnames()
    tb.close()
    print(colnames)

def inspectMatColumns(mat):
    mat_data=loadmat(mat)
    print(mat_data.keys())
    print(mat_data['y'].shape)
    print(mat_data['y'])

def inspectVisibilitiesColumns(ms):
    tb.open(ms)
    visibilities=tb.getcol('DATA')
    tb.close()
    print(visibilities.shape)

# Copy visibilities to data
def copyVisibilitiesToData(ms, mat):
    tb.open(ms, nomodify=False)
    data = tb.getcol('DATA')
    mask = tb.getcol('FLAG')[0, 0, :] == False
    data[:, 0, mask] = loadmat(mat)['y'].flatten()
    tb.putcol('DATA', data)
    tb.close()


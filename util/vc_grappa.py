import sys
sys.path.insert(1, '../')
import numpy as np
import matplotlib.pyplot as plt
from util.coil import *
from util.fft import *
from util.grappa import * 

def vc_grappa(dataR, calib, R, kh = 4, kw = 5):
    kspace = dataR
    vc_kspace = conjugate(kspace)
    vc_calib = conjugate(calib)
    kspace = np.concatenate((kspace, vc_kspace), axis=-1)
    calib = np.concatenate((calib, vc_calib), axis=-1)
    recon = grappa(kspace, calib, R, kh, kw)
    return recon 
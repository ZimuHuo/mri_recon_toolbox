'''
Author: Zimu Huo
Date: 12.23.2020
'''
# ‘C’ row-major (C-style), ‘F’ column-major (Fortran-style)
# I used matlab to read the .m file and export it, i forgot matlab is in c major
# So the order = 'F' is necessary when reading the current files 
# I guess the final image is also transposed
import numpy as np
import os 
import mapvbvd
def getCrd(filepath):
    '''
    read the k space location file, it comes as x_data1, y_data1, x_data2, y_data2...
    '''
    filename = filepath
    crd = np.fromfile(filename, dtype='float32')
    crd = np.reshape(crd,[2,13392],order='F') 
    return crd

def getData(filepath):
    '''
    read the MRI data, in this case, there is only one trajectory from one coil
    the data comes in as data1_real, data1_img, data2_real, data2_img...
    '''
    filename = filepath
    file = np.fromfile(filename, dtype='float32')
    data = np.zeros(744*18, dtype=complex)
    data = file[0::2]+1j*file[1::2]
    data = np.reshape(data,([1,744*18]),order='F')
 
    return data

def getDCF():
    filename = '../lib/resource/data/spiral_1slice_1cha/dcf.dat'
    file = np.fromfile(filename, dtype='float32')
    data = np.zeros(744*18,np.int32)
    data = file[0::1]
    dcf = np.reshape(data,([1,744*18]),order='F')
    return dcf[0,:]

def checkFile(fileDir):
    return os.path.isfile(fileDir) 

def getCoilData(filepath):

    filename = filepath
    twix = mapvbvd.mapVBVD(filename)
    data = twix.image['']
    d = np.squeeze(np.array(data))
    return np.swapaxes(d, 1, 2)
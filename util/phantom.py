import numpy as np
from pdf2image import convert_from_path
import numpy.matlib
from util.fft import *

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def sms(fov, coil):
    b = brain(fov, coil, mode = 'f')
    s = scale(shepp_logan(fov,coil, mode = 'f'), (np.min(b), np.max(b)))
    sens = generate_birdcage_sensitivities(fov,coil)
    bs = b * sens
    ss = s * sens
    rawImage = np.zeros([fov, fov, coil, 2], dtype = complex)
    rawImage[...,0] = bs
    rawImage[...,1] = ss
    return rawImage
    
import scipy.io
def get_tissue_images(slices):
    tissuetype = ['graymatter', 'deep_graymatter', 'whitematter', 'csf']
    T2 = [110, 100, 60, 1500]
    T2s = [40, 45, 50, 1000]
    mat = scipy.io.loadmat('../../lib/resource/data/tissue_images/tissue_images.mat')
    tissues = mat.get("tissue_images")[:,:,slices,:]
    return np.squeeze(tissues), tissuetype

def brain_tissue(slices, coils):
    tissues,tissuetype = get_tissue_images(slices)
    if (coils== 1): return tissues,tissuetype, T2, T2s
    tissues = np.repeat(tissues[:, :,  np.newaxis,:], coils, axis=-2)
    coils = generate_birdcage_sensitivities(matrix_size = 222,number_of_coils = coils)
    tissues = tissues *  np.repeat(coils[...,np.newaxis], 4, axis = -1)
    return tissues,tissuetype
    
def brain(fov, coil = 1, mode = 'F'):
    image = np.array(convert_from_path('../lib/Brain.pdf', size = (fov, fov))[0])
    image = np.asarray(np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]), dtype = complex)
    if (coil== 1): return image
    im = np.repeat(image[:, :, np.newaxis], coil, axis=2)
    s = im * generate_birdcage_sensitivities(matrix_size = fov,number_of_coils = coil)
    if mode == 'F': return fft2c(s)
    else: return s


def shepp_logan(fov, coil = 1, mode = 'F'):
    im = np.repeat(shepp_logan_phantom([fov,fov])[:, :, np.newaxis], coil, axis=2)
    s = im * generate_birdcage_sensitivities(matrix_size = fov,number_of_coils = coil)
    if (coil== 1):  return shepp_logan_phantom([fov,fov])
    if mode == 'F': return fft2c(s)
    else: return s
    
def generate_birdcage_sensitivities(matrix_size = 256, number_of_coils = 8, relative_radius = 1.5, normalize=True):
    """ Generates birdcage coil sensitivites.
    :param matrix_size: size of imaging matrix in pixels (default ``256``)
    :param number_of_coils: Number of simulated coils (default ``8``)
    :param relative_radius: Relative radius of birdcage (default ``1.5``)
    This function is heavily inspired by the mri_birdcage.m Matlab script in
    Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/
    """

    out = np.zeros((number_of_coils,matrix_size,matrix_size),dtype=np.complex128)
    for c in range(0,number_of_coils):
        coilx = relative_radius*np.cos(c*(2*np.pi/number_of_coils))
        coily = relative_radius*np.sin(c*(2*np.pi/number_of_coils))
        coil_phase = -c*(2*np.pi/number_of_coils)

        for y in range(0,matrix_size):
            y_co = float(y-matrix_size/2)/float(matrix_size/2)-coily
            for x in range(0,matrix_size):
                x_co = float(x-matrix_size/2)/float(matrix_size/2)-coilx
                rr = np.sqrt(x_co**2+y_co**2)
                phi = np.arctan2(x_co, -y_co) + coil_phase
                out[c,y,x] =  (1/rr) * np.exp(1j*phi)

    if normalize:
         rss = np.squeeze(np.sqrt(np.sum(abs(out) ** 2, 0)))
         out = out / np.tile(rss,(number_of_coils,1,1))
    
    out = np.moveaxis(out, 0 , -1)

    return out



#copied from sigpy
def shepp_logan_phantom(fov):
    """Generates a Shepp Logan phantom with a given shape and dtype.
    Args:
        shape (tuple of ints): shape, can be of length 2 or 3.
        dtype (Dtype): data type.
    Returns:
        array.
    """
    image = phantom(fov, sl_amps, sl_scales, sl_offsets, sl_angles, dtype = complex)
    return image



sl_amps = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sl_scales = [[.6900, .920, .810],  # white big
             [.6624, .874, .780],  # gray big
             [.1100, .310, .220],  # right black
             [.1600, .410, .280],  # left black
             [.2100, .250, .410],  # gray center blob
             [.0460, .046, .050],
             [.0460, .046, .050],
             [.0460, .046, .050],  # left small dot
             [.0230, .023, .020],  # mid small dot
             [.0230, .023, .020]]

sl_offsets = [[0., 0., 0],
              [0., -.0184, 0],
              [.22, 0., 0],
              [-.22, 0., 0],
              [0., .35, -.15],
              [0., .1, .25],
              [0., -.1, .25],
              [-.08, -.605, 0],
              [0., -.606, 0],
              [.06, -.605, 0]]

sl_angles = [[0, 0, 0],
             [0, 0, 0],
             [-18, 0, 10],
             [18, 0, 10],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]


def phantom(shape, amps, scales, offsets, angles, dtype):
    """
    Generate a cube of given shape using a list of ellipsoid
    parameters.
    """

    if len(shape) == 2:

        ndim = 2
        shape = (1, shape[-2], shape[-1])

    elif len(shape) == 3:

        ndim = 3

    else:

        raise ValueError('Incorrect dimension')

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[-(shape[-3] // 2):((shape[-3] + 1) // 2),
                       -(shape[-2] // 2):((shape[-2] + 1) // 2),
                       -(shape[-1] // 2):((shape[-1] + 1) // 2)]

    coords = np.stack((x.ravel() / shape[-1] * 2,
                       y.ravel() / shape[-2] * 2,
                       z.ravel() / shape[-3] * 2))

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):

        ellipsoid(amp, scale, offset, angle, coords, out)

    if ndim == 2:

        return out[0, :, :]

    else:

        return out


def ellipsoid(amp, scale, offset, angle, coords, out):
    """
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new
    one.
    """
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / \
        np.reshape(scale, (3, 1))

    r2 = np.sum(coords ** 2, axis=0).reshape(out.shape)

    out[r2 <= 1] += amp


def rotation_matrix(angle):
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
              -spsi * sphi + ctheta * cphi * cpsi,
              cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.array(alpha)

def coilmap(
        size = 64, ncoils=4, array_cent=None, coil_width=8, n_rings=None, phi=0):
    """ Apply simulated sensitivity maps. Based on a script by Florian Knoll.

        Args:
            size (tuple): Size of the image array for the sensitivity coils.
            nc_range (int, default: 8): Number of coils to simulate.
            array_cent (tuple, default: 0): Location of the center of the coil
                array
            coil_width (double, default: 2): Parameter governing the width of
                the coil, multiplied by actual image dimension.
            n_rings (int, default: ncoils // 4): Number of rings for a
                cylindrical hardware set-up.
            phi (double, default: 0): Parameter for rotating coil geometry.

        Returns:
            coil_array (array_like): An array of dimensions (ncoils (N)),
                specifying spatially-varying sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if (len(size) > 2):
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if (len(size) > 2):
        zz, yy, xx = np.meshgrid(range(size[2]), range(size[1]),
                                 range(size[0]), indexing='ij')
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]),
                             indexing='ij')

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if (len(size) > 2):
                theta = np.radians((i-1)*360/(ncoils + n_rings) + phi)
            else:
                theta = np.radians((i-1)*360/ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0]/2
            y0[i] = c_rad * np.sin(theta) + size[1]/2
            if (len(size) > 2):
                z0[i] = (size[2]/(n_rings+1)) * (i//n_rings)
                smap.append(np.exp(-1*((xx-x0[i])**2 + (yy-y0[i])**2 +
                                       (zz-z0[i])**2) / (2*c_width)))
            else:
                smap.append(np.exp(-1*((xx-x0[i])**2 + (yy-y0[i])**2) /
                                   (2*c_width)))
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if (len(size) > 2):
            smap = np.exp(-1*((xx-x0)**2 + (yy-y0)**2 +
                              (zz-z0)**2) / (2*c_width))
        else:
            smap = np.exp(-1*((xx-x0)**2 + (yy-y0)**2) / (2*c_width))

    side_mat = np.arange(int(size[0]//2)-20, 1, -1)
    side_mat = (np.reshape(side_mat, (1,) + side_mat.shape) *
                np.ones(shape=(size[1], 1)))
    cent_zeros = np.zeros(shape=(size[1], size[0]-side_mat.shape[1]*2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 5
    if (len(size) > 2):
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i*1j*ph*np.pi/180)
    smap = np.moveaxis(smap, 0, -1)
    return smap


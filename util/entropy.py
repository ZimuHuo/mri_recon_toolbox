from util.fft import *
from util.epi import *
from util.rank import *
import cv2 
def entropy(data):
    if len(data.shape) > 2:
        image = np.sum(np.abs(data)**2, -1)
    else:
        image = np.abs(data)**2
    b = np.sqrt(image)/np.sum(image)
    return np.abs(np.sum(b/np.log(b)))
    
def object_projection_from_central_line(ref):
    ref = ref[ref.shape[0]//2-1]
    obj = (np.sqrt(np.sum(np.abs(ifft(ref))**2,1)))
    obj *= 1/obj.max()
    true_obj = np.copy(obj)
    obj[obj < 1/5] = 0
    obj = scipy.signal.medfilt(obj, 11)
    return true_obj


def epi_phase_correction_even_odd(data, tvec):
    [ny, nx, nc] = data.shape
    tvecm = np.exp(1j*tvec).reshape(-1,1)
    even = np.zeros(data.shape, dtype = complex)
    odd = np.zeros(data.shape, dtype = complex)
    even[::2] = data[::2]
    odd[1::2] = data[1::2]
    dataflag = np.zeros([nx, 1])
    dataflag[1::2] = 1
    odd = apply_epi_phase(odd, tvecm)
    ghost_data = ifft2c(even) + ifft2c(odd)
    return ghost_data

def apply_epi_phase(data, tvecm):
    recon = np.zeros(data.shape, dtype = complex)
    tvec = np.ones(data.shape, dtype = complex)
    tvec[1::2,...] = np.tile(tvecm.reshape(-1,1), data.shape[2])
    data = ifft(data,1)
    recon = data * tvec
    recon = ifft(recon,0)
    return fft2c(recon) 


def epi_entropy_cost(p, data):
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    ghost_data = epi_phase_correction_even_odd(data,tvec)
    return entropy(ghost_data)

def epi_phasecorrection_entropy(data, grid = 50, output ="partial"):
    cur = entropy(ifft2c(data))
    recon = np.zeros(data.shape, dtype = complex)
    p = scipy.optimize.brute(epi_entropy_cost,((-5,5),(-1,1)), args=(data,), Ns = grid, workers = 16)
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    recon = epi_phase_correction_even_odd(data, tvec)
    if output == "full":
        return fft2c(recon), np.exp(1j*tvec).reshape(-1,1) , tvec
    else:
        return fft2c(recon)
    
def epi_crosscorr_cost(p, data):
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    ghost_data = np.abs(epi_phase_correction_even_odd(data,tvec))
    return np.sum(np.abs(ghost_data * np.roll(ghost_data, -ny//2, 0)))

def epi_phasecorrection_crosscorr(data, grid = 50, output ="partial"):
    recon = np.zeros(data.shape, dtype = complex)
    amin = 0
    amax = 10
    astep = (amax-amin)/grid
    bmin = -1
    bmax = 1
    bstep = (bmax-bmin)/grid
    rrange = (slice(amin,amax,astep), slice(bmin, bmax, bstep))
    p = scipy.optimize.brute(epi_crosscorr_cost,rrange, args=(data,), Ns = grid, workers = 16)
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    recon = epi_phase_correction_even_odd(data, tvec)
    if output == "full":
        return fft2c(recon), np.exp(1j*tvec).reshape(-1,1) , tvec
    else:
        return fft2c(recon)
def sake_forward(data, k):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-k+1)*(nx-k+1), k * k * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[idx, :] = data[:,y:y+k, x:x+k].reshape(1,-1)
            idx += 1
    return mat
def epi_lowrank_cost(p, data):
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    ghost_data = fft2c(epi_phase_correction_even_odd(data,tvec))
    mat = sake_forward(ghost_data, k = 3)
    U, S, VT = np.linalg.svd(mat,full_matrices=False)
    return np.sum(S[1:])

def epi_phasecorrection_lowrank(data, grid = 50, output ="partial"):
    recon = np.zeros(data.shape, dtype = complex)
    p = scipy.optimize.brute(epi_lowrank_cost,((0,10),(-1,1)), args=(data,), Ns = grid, workers = 16)
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    recon = epi_phase_correction_even_odd(data, tvec)
    if output == "full":
        return fft2c(recon), np.exp(1j*tvec).reshape(-1,1) , tvec
    else:
        return fft2c(recon)
    
def epi_ghostobject_cost(p, data):
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    ghost_data = np.abs(epi_phase_correction_even_odd(data,tvec))
    ratio = (ghost_data / np.roll(ghost_data, -ny//2, 0))
    filtered = ndimage.median_filter(np.abs(ratio), size = 3) 
    return 1/np.sum(filtered)

def epi_phasecorrection_ghostobject(data, grid = 50, output ="partial"):
    recon = np.zeros(data.shape, dtype = complex)
    amin = 0
    amax = 10
    astep = (amax-amin)/grid
    bmin = -1
    bmax = 1
    bstep = (bmax-bmin)/grid
    rrange = (slice(amin,amax,astep), slice(bmin, bmax, bstep))
    p = scipy.optimize.brute(epi_crosscorr_cost,rrange, args=(data,), Ns = grid, workers = 16)
    [ny, nx, nc] = data.shape
    x = np.linspace(-0.5, 0.5, nx)
    obj = object_projection_from_central_line(data)
    model = p[0] * x + p[1]
    tvec = polyfit(x, model, w = obj, order = 1)
    recon = epi_phase_correction_even_odd(data, tvec)
    if output == "full":
        return fft2c(recon), np.exp(1j*tvec).reshape(-1,1) , tvec
    else:
        return fft2c(recon)
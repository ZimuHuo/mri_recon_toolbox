from tqdm.notebook import tqdm
import numpy as np
def gaussian_noise(shape, L = None, mu = 0, sigma = 1):
    [ny, nx, nc] = shape
    n = np.zeros([ny * nx, nc], dtype = complex)
    n.real = np.random.normal(mu, sigma, ny*nx*nc).reshape(ny * nx, nc)
    n.imag = np.random.normal(mu, sigma, ny*nx*nc).reshape(ny * nx, nc)
    if L is not None: 
        n = n@L
    return n.reshape(ny, nx, nc)


def pseudo_replica_grappa_snr(noise, data, acs, R, nt = 50):
    ny , nx, nc = data.shape
    ny = ny * R
    phi = 1/(2*noise.shape[0])* (noise.T.conj() @ noise)
    L = np.linalg.cholesky(phi)
    pseudodata = np.zeros([ny, nx, nt], dtype = complex)
    for rep in tqdm(range(nt)):
        tmpdata = data + gaussian_noise(data.shape, L)
        pseudodata[...,rep] = grappa(tmpdata, acs, R)
    pseudodata = np.abs(pseudodata)
    stdmap = np.std(pseudodata, -1)
    snrmap = np.mean(pseudodata, -1)/ stdmap / np.sqrt(R)
    return snrmap
from venv import create

import numpy as np

def create_dft_matrix(N, inverse=False):
    # row vector mapping (1, N)
    x = np.arange(N).reshape((1, N))

    # column vector mapping (N, 1)
    u = np.arange(N).reshape((N, 1))

    # N * N matrix containing all possible frequencies
    matrix_product = x * u

    # matrix containing all angles
    theta_matrix = 1 if inverse else -1
    theta_matrix *= 2 * np.pi * matrix_product / N

    # the discrete fourier transform matrix
    return np.cos(theta_matrix) + 1j * np.sin(theta_matrix)

def dft2d(image):
    H, W = image.shape

    # dft for height
    W_H = create_dft_matrix(H)

    # dft for width
    W_W = create_dft_matrix(W)

    # the spectrum from image
    spectrum = W_H @ image @ W_W
    return spectrum

def idft2d(spectrum):
    H, W = spectrum.shape

    # inverse dft for height
    inv_W_H = create_dft_matrix(H, inverse=True)

    # inverse dft for weight
    inv_W_W = create_dft_matrix(W, inverse=True)

    # the reconstructed image using fourier transform
    image = inv_W_H @ spectrum @ inv_W_W / (H * W)
    return np.real(image)

def fft_shift(spectrum):
    # the fast fourier transform moving the corner frequencies in the middle of the spectrum
    # centres the spectrum around the origin

    H, W = spectrum.shape

    # rolls the spectrum values by half the height
    shifted = np.roll(spectrum, H // 2, axis=0)

    # rolls the spectrum values by half the width
    shifted = np.roll(shifted, W // 2, axis=1)

    return shifted

def rfft2d(image):
    # real 2d fast fourier transform
    # reduces the size in half (hermitian symmetry)
    # the second half of the spectrum is the first one reversed

    H, W = image.shape
    return dft2d(image)[:, :W//2 + 1]


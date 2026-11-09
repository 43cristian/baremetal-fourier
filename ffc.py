import numpy as np
from spectral import dft2d, idft2d, fft_shift, rfft2d

def ifft_shift(spectrum):
    # inverse fast fourier transform shift
    # moves the central frequencies back to the corners

    H, W = spectrum.shape
    shifted = np.roll(spectrum, -(H//2), axis=0)
    shifted = np.roll(shifted, -(W//2), axis=1)
    return shifted

def apply_frequency_mask(spectrum, radius, high_pass=False):
    # applies a geometric filter on the spectrum
    # low pass = keeps centre -> keeps shapes, loses details
    # high pass = keeps exterior -> keeps outlines

    H, W = spectrum.shape
    center_y, center_x = H // 2, W // 2

    # coordinate grid to compute distance from the center
    Y, X = np.ogrid[:H, :W]
    distance = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)

    filtered_spectrum = spectrum.copy()

    if high_pass:
        filtered_spectrum[distance < radius] = 0 + 0j
    else:
        filtered_spectrum[distance > radius] = 0 + 0j

    return filtered_spectrum

def percentage_mask(spectrum, drop_percentage):
    # drops % of frequencies with lowest energy
    # (compression)

    magnitude = np.abs(spectrum)

    treshold = np.percentile(magnitude, drop_percentage)

    filtered_spectrum = spectrum.copy()
    filtered_spectrum[magnitude < treshold] = 0 + 0j

    return filtered_spectrum

class FourierUnit:
    def __init__(self, in_channels=1, out_channels=1):
        # one for real values, one for imaginary values
        self.dim = in_channels * 2

        # random weights and biases
        # simulates a trained network
        self.W = np.random.randn(out_channels * 2, self.dim) * 0.1
        self.b = np.zeros((out_channels * 2, 1, 1))

    def forward(self, x):
        C, H, W_dim = x.shape

        spectrum = np.zeros((C, H, W_dim), dtype=np.complex128)
        for c in range(C):
            spectrum[c] = dft2d(x[c])

        # splitting real/imag
        real = np.real(spectrum)
        imag = np.imag(spectrum)

        cont_spectrum = np.concatenate([real, imag], axis=0)

        # 1x1 convolution
        C_conc = cont_spectrum.shape[0]
        flat_spectrum = cont_spectrum.reshape(C_conc, -1)
        proc_spectrum = self.W @ flat_spectrum
        proc_spectrum = proc_spectrum.reshape(self.W.shape[0], H, W_dim) + self.b

        # ReLU
        activated_spectrum = np.maximum(0, proc_spectrum)

        # splitting real/imag again
        half = activated_spectrum.shape[0] // 2
        new_real = activated_spectrum[:half, :, :]
        new_imag = activated_spectrum[half:, :, :]

        new_spectrum = new_real + 1j * new_imag

        out = np.zeros((half, H, W_dim), dtype=np.complex128)
        for c in range(half):
            out[c] = idft2d(new_spectrum[c])

        return np.real(out)
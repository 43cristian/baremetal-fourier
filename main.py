from typing import final

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ffc import apply_frequency_mask, ifft_shift, percentage_mask, FourierUnit
from spectral import dft2d, idft2d, fft_shift, rfft2d

def main():
    img_path = "images/landscape.jpg"
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float64) / 255.0

    H, W = img_array.shape
    print(f"processed image size: {H}x{W}")
    spectrum = dft2d(img_array)
    reconstructed_image = idft2d(spectrum)

    max_error = np.max(np.abs(img_array - reconstructed_image))
    print(f"max error: {max_error:.2e}")

    central_spectrum = fft_shift(spectrum)
    magnitude = 20 * np.log(np.abs(central_spectrum) + 1e-8)

    orig_vmin = magnitude.min()
    orig_vmax = magnitude.max()

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 5, 1)
    plt.title("1. Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 5, 2)
    plt.title("2. Fourier Spectrum")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 5, 3)
    plt.title("3. Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')

    filtered_central_spectrum = apply_frequency_mask(central_spectrum, radius=50, high_pass=False)
    filtered_spectrum = ifft_shift(filtered_central_spectrum)
    filtered_image = idft2d(filtered_spectrum)

    plt.subplot(2, 5, 4)
    plt.title("4. Low Pass Image")
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    magnitude = 20 * np.log(np.abs(fft_shift(filtered_spectrum)) + 1e-8)

    plt.subplot(2, 5, 6)
    plt.title("6. Low Pass Spectrum")
    plt.imshow(magnitude, cmap='gray', vmin=orig_vmin, vmax=orig_vmax)
    plt.axis('off')

    filtered_central_spectrum = apply_frequency_mask(central_spectrum, radius=50, high_pass=True)
    filtered_spectrum = ifft_shift(filtered_central_spectrum)
    filtered_image = idft2d(filtered_spectrum)

    plt.subplot(2, 5, 5)
    plt.title("5. High Pass Image")
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    magnitude = 20 * np.log(np.abs(fft_shift(filtered_spectrum)) + 1e-8)

    plt.subplot(2, 5, 7)
    plt.title("7. High Pass Spectrum")
    plt.imshow(magnitude, cmap='gray', vmin=orig_vmin, vmax=orig_vmax)
    plt.axis('off')

    compressed_spectrum = percentage_mask(central_spectrum, drop_percentage=70)
    ispectrum = ifft_shift(compressed_spectrum)
    compressed_image = idft2d(ispectrum)

    plt.subplot(2, 5, 8)
    plt.title("8. Compressed Image")
    plt.imshow(compressed_image, cmap='gray')
    plt.axis('off')

    magnitude = 20 * np.log(np.abs(compressed_spectrum) + 1e-8)

    plt.subplot(2, 5, 9)
    plt.title("9. Compressed Spectrum")
    plt.imshow(magnitude, cmap='gray', vmin=orig_vmin, vmax=orig_vmax)
    plt.axis('off')

    image_tensor = img_array.reshape(1, img_array.shape[0], img_array.shape[1])
    fu_layer = FourierUnit(in_channels=1, out_channels=1)
    processed_image = fu_layer.forward(image_tensor)
    final_image = processed_image.squeeze(0)

    plt.subplot(2, 5, 10)
    plt.title("10. Fourier Unit Output")
    plt.imshow(final_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

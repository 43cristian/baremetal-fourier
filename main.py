import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.title("1. Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Fourier Spectrum")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

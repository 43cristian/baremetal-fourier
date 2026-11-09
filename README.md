# 🚀 Bare-Metal FFC: Fast Fourier Convolutions in NumPy

A bare-metal, dependency-free NumPy implementation of 2D Discrete Fourier Transforms and Fast Fourier Convolutions (FFC) to explore global receptive fields in computer vision.

This project serves as the mathematical foundation and proof-of-concept for a Bachelor's thesis focused on optimizing **Image Inpainting algorithms (e.g., LaMa architecture)** for highly efficient execution on mobile devices (Edge Computing).

By stripping away the "black-box" abstractions of modern deep learning frameworks (like PyTorch or TensorFlow), this repository demonstrates the first-principles mechanics of frequency manipulation and global context aggregation.

## 🧠 Key Concepts Demonstrated

1. **Global Receptive Field:**
   Unlike classical spatial convolutions, the Fourier Transform allows the network to "see" the entire image simultaneously. This project visually demonstrates how altering a single frequency domain parameter affects the image at a global scale.

2. **Spectral Compression for Mobile Devices:**
   By dropping low-energy frequencies (simulating spectral cropping), we demonstrate how mobile-optimized networks can massively reduce RAM and battery consumption while preserving the visual coherence of the scene.

3. **Frequency-Domain Learning (`FourierUnit`):**
   Demonstrates how a neural network applies trainable weights ($W$) over complex numbers to dynamically learn which frequencies to amplify and which to suppress—the exact mathematical equivalent of a $1 \times 1$ convolution in the spectral space.

## 📂 Project Structure

* **`spectral.py`** - The core mathematical engine. Contains the complete, custom implementation of the 2D Discrete Fourier Transform (`dft2d`) and its Inverse (`idft2d`), without relying on pre-compiled FFT signal processing packages.

* **`ffc.py`** - Neural logic and spectral manipulation. Includes geometric filters (`apply_frequency_mask`), compression algorithms (`percentage_mask`), and the core neural component: the `FourierUnit` class.

* **`main.py`** - The execution and visualization pipeline. Loads a test image and generates a comprehensive 10-plot Matplotlib dashboard that breaks down every mathematical step of the algorithm.

## 🛠️ Installation and Execution

Because this project is built to be minimalist and purely mathematical, dependencies are kept to an absolute minimum.

```
# 1. Clone the repository
git clone https://github.com/43cristian/bare-metal-ffc.git
cd bare-metal-ffc

# 2. Install dependencies (preferably in a virtual environment)
pip install numpy matplotlib pillow

# 3. Run the visual demonstration
python main.py

```

*Note: Ensure you have a valid test image (e.g., `landscape.jpg`) in the `images/` directory before running the script.*

## 📊 Visualizing the Output

Running the main script generates a visual dashboard detailing the following logical steps:

1. **Original Image & Frequency Spectrum:** Visual representation of magnitudes (illumination in the center, sharp details towards the edges).

2. **Bare-Metal Reconstruction:** Proof that the custom Direct and Inverse Transform algorithms work flawlessly (Error $\approx 0$).

3. **Low-Pass / High-Pass Filtering:** Extracting base shapes and illumination (low) versus extracting sharp textures and outlines (high).

4. **Percentage Compression:** Yields a highly recognizable image even after intentionally dropping 70-90% of the spectral data (the essential proof for hardware-limited inference).

5. **FourierUnit Output:** Simulates the very first "thought process" of a neural network by applying a matrix multiplication ($1 \times 1$ conv) directly over the frequencies.

## 🎓 Academic Context (Bachelor's Thesis)

This module acts as the theoretical foundation for my Bachelor's thesis. The immediate next step in the research is to combine this global `FourierUnit` with a standard spatial convolution ($3 \times 3$) to construct a complete **FFC_block**. This hybrid architecture will successfully merge global reasoning (resolved spectrally) with local texture refinement (resolved spatially), yielding an ultra-efficient inpainting model tailored for mobile hardware.
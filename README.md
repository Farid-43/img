# Image Processing Code Examples

A collection of Digital Image Processing algorithms implemented in Python using OpenCV, NumPy, and Matplotlib.

---

## 📁 Repository Overview

| # | Script | Category | Description |
|---|---|---|---|
| **01** | `1_log.py` | Spatial Domain | Logarithmic Transformation |
| **02** | `2_gamma.py` | Spatial Domain | Power-Law (Gamma) Transformation |
| **03** | `3_thres.py` | Spatial Domain | Intensity Thresholding |
| **04** | `4. hist equal.py` | Histogram | Gray Level Histogram Equalization |
| **05** | `5. hist qualization rgb.py` | Histogram | Color (RGB) Histogram Equalization |
| **06** | `6. hist matching.py` | Histogram | Histogram Matching (Specification) |
| **07** | `7. hist inverse.py` | Histogram | Histogram Inversion / PDF Matching |
| **08** | `8. guassian.py` | Spatial Filtering | 2D Gaussian Smoothing Filter |
| **09** | `9. laplacian shobel.py` | Spatial Filtering | Laplacian & Sobel Edge Detection |
| **10** | `10. IDPF.py` | Frequency Domain | Ideal Lowpass / Notch Reject Filtering (2D FFT) |
| **11** | `11. butterworth notch.py` | Frequency Domain | Butterworth Notch Reject Filter |
| **12** | `12. butterworth lowpass.py` | Frequency Domain | Butterworth Lowpass Filter |
| **13** | `13. butterworth highpass.py` | Frequency Domain | Butterworth Highpass Filter |
| **14** | `14. gaussian notch.py` | Frequency Domain | Gaussian Notch Reject Filter |
| **15** | `15. gaussian lowpass.py` | Frequency Domain | Gaussian Lowpass Filter |
| **16** | `16. gaussian highpass.py` | Frequency Domain | Gaussian Highpass Filter |

---

## 🛠️ Requirements & Setup

Ensure Python 3.x is installed along with the required libraries:

```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Execution

Run any script directly from the project root directory:

```bash
python 1_log.py
python "10. IDPF.py"
```

> **Note:** Sample images are located inside the `img/` folder.

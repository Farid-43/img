# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the given problem data:
# - Histogram equalized image Y
# - Given PDF of the original image X: PDF(X)
image_Y = cv2.imread(r"img/equalized_Y.png", cv2.IMREAD_GRAYSCALE)

pdf_X = np.load(r"img/pdf_X.npy")

height = image_Y.shape[0]
width = image_Y.shape[1]

# Functions for PDF and CDF
def get_pdf(channel):
    h = channel.shape[0]
    w = channel.shape[1]
    pdf = np.zeros(256, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            pdf[channel[i, j]] += 1
    pdf /= (h * w)
    return pdf

def get_cdf(pdf):
    cdf = np.zeros_like(pdf)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf

# 2. Compute CDF of original image X
cdf_X = get_cdf(pdf_X)

# 3. Compute forward transformation table T(r) = round(255 * CDF_X(r))
T = np.zeros(256, dtype=np.uint8)
for r in range(256):
    T[r] = round(cdf_X[r] * 255)

# 4. Compute inverse mapping:
inv_map = np.zeros(256, dtype=np.uint8)
for s in range(256):
    k = s
    v = T[k]
    inv_map[v] = k

# 5. Reconstruct the original image X
recons_image = np.zeros_like(image_Y)
for i in range(height):
    for j in range(width):  
        recons_image[i, j] = inv_map[image_Y[i, j]] # INVERSE Mapping
        # recons_image[i, j] = T[image_Y[i, j]] # FORWARD Mapping (USED FOR HISTOGRAM EQUALIZATION)

# 6. Compute PDFs and CDFs for Reconstructed and Equalized (Output) images
pdf_recons = get_pdf(recons_image)
cdf_recons = get_cdf(pdf_recons)

pdf_Y = get_pdf(image_Y)
cdf_Y = get_cdf(pdf_Y)

# 7. Plotting as per the required output format (2 rows x 3 columns)
plt.figure(figsize=(14, 8))

# --- Row 1 ---
# Reconstructed image
plt.subplot(2, 3, 1)
plt.imshow(recons_image, cmap="gray")
plt.title("Recons")
plt.axis("off")

# PDF of given and reconstructed image
plt.subplot(2, 3, 2)
plt.plot(pdf_X, color="red", label="Input")
plt.plot(pdf_recons, color="green", label="Recons")
plt.title("PDF")
plt.legend()

# CDF of given and reconstructed image
plt.subplot(2, 3, 3)
plt.plot(cdf_X, color="red", label="Input")
plt.plot(cdf_recons, color="green", label="Recons")
plt.title("CDF")
plt.legend()

# --- Row 2 ---
# Output Image (Y)
plt.subplot(2, 3, 4)
plt.imshow(image_Y, cmap="gray")
plt.title("Output")
plt.axis("off")

# PDF of output image
plt.subplot(2, 3, 5)
plt.plot(pdf_Y, color="red")
plt.title("PDF")

# CDF of output image
plt.subplot(2, 3, 6)
plt.plot(cdf_Y, color="red")
plt.title("CDF")

plt.tight_layout()
plt.savefig("hist_inverse_output.png", dpi=300, bbox_inches="tight")
plt.show()

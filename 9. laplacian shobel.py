# 9. laplacian shobel.py
# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"img/retina_2.png", cv2.IMREAD_GRAYSCALE)
height = image.shape[0]
width = image.shape[1]


# 2D Convolution function
def conv2d(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    # padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    # padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(kernel * padded_image[i:i+k_h, j:j+k_w])
            
    return output


# --- 1. Laplacian Filter & Sharpening ---
# Laplacian kernel (with positive center +4)
laplacian_kernel = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Optional 8-neighbor Laplacian kernel:
# laplacian_kernel = np.array([
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1]
# ], dtype=np.float32)

# laplacian sharpening : I + c * conv2d(I, laplacian_kernel)
laplacian_edge = conv2d(image, laplacian_kernel)
c = 1.0
laplacian_sharpened = image.astype(np.float32) + c * laplacian_edge

laplacian_sharpened = cv2.normalize(laplacian_sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
laplacian_display = cv2.normalize(laplacian_edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# --- 2. Sobel Filter ---
# Sobel X (vertical edges / horizontal gradient)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel Y (horizontal edges / vertical gradient)
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Apply Sobel filters
gx = conv2d(image, sobel_x)
gy = conv2d(image, sobel_y)

# Sobel Combined Gradient Magnitude: sqrt(gx^2 + gy^2)
sobel_combined = np.sqrt(gx**2 + gy**2)
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

gx_display = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gy_display = cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# --- Plotting Results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(laplacian_display, cmap="gray", vmin=0, vmax=255)
plt.title("Laplacian (Edges)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(laplacian_sharpened, cmap="gray", vmin=0, vmax=255)
plt.title("Laplacian Sharpened")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gx_display, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gy_display, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sobel_combined, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel Combined")
plt.axis("off")

plt.tight_layout()
plt.savefig("out.png", dpi=300, bbox_inches="tight")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# =========================================================
# 1. GAUSSIAN KERNEL
# =========================================================

def get_gauss_kernel(sigma, size):
    kernel = np.zeros((size, size))
    center = size // 2

    total = 0.0

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            value = math.exp(
                -(x * x + y * y) / (2 * sigma * sigma)
            )

            kernel[i][j] = value
            total = total + value

    # Normalize Gaussian kernel so that sum = 1
    for i in range(size):
        for j in range(size):
            kernel[i][j] = kernel[i][j] / total

    return kernel


# =========================================================
# 2. FIRST DERIVATIVE OF GAUSSIAN
# =========================================================

def get_gauss_derivative_kernel(sigma, size, direction="x"):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            gaussian = math.exp(
                -(x * x + y * y) / (2 * sigma * sigma)
            )

            if direction == "x":
                value = -(x / (sigma * sigma)) * gaussian  

            elif direction == "y":
                value = -(y / (sigma * sigma)) * gaussian

            else:
                raise ValueError("direction must be 'x' or 'y'")

            kernel[i][j] = value

    return kernel


# =========================================================
# 3. LAPLACIAN OF GAUSSIAN (LoG)
# =========================================================

def get_log_kernel(sigma, size):
    #equation: LoG = (r^2 - 2σ^2) / σ^4 * exp(-r^2 / 2σ^2)
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            r2 = x * x + y * y

            value = (
                (r2 - 2 * sigma * sigma) 
                / (sigma ** 4)
                * math.exp(-r2 / (2 * sigma * sigma))
            )

            kernel[i][j] = value

    return kernel


# =========================================================
# 4. SOBEL KERNEL
# =========================================================

def get_sobel_kernel(direction="x"):

    if direction == "x":

        kernel = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ], dtype=np.float32)

    elif direction == "y":

        kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)

    else:
        raise ValueError("direction must be 'x' or 'y'")

    return kernel


# =========================================================
# CONVOLUTION FUNCTION
# =========================================================

def convolve(image, kernel):

    img_height = image.shape[0]
    img_width = image.shape[1]

    k_size = kernel.shape[0]
    pad = k_size // 2

    # Zero-padded image
    padded = np.zeros(
        (img_height + 2 * pad, img_width + 2 * pad),
        dtype=np.float32
    )

    for i in range(img_height):
        for j in range(img_width):
            padded[i + pad][j + pad] = image[i][j]

    # Raw floating-point output
    output = np.zeros((img_height, img_width),dtype=np.float32)

    # Convolution
    for i in range(img_height):
        for j in range(img_width):

            acc = 0.0 

            for ki in range(k_size):
                for kj in range(k_size):

                    acc += (
                        padded[i + ki][j + kj]
                        * kernel[k_size - 1 - ki][k_size - 1 - kj]
                    )

            # DO NOT CLIP HERE
            output[i][j] = acc

    return output


# =========================================================
# EDGE OUTPUT PROCESSING
# =========================================================

def prepare_edge_output(output):
    output = np.abs(output)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return output


# =========================================================
# READ IMAGE
# =========================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "img", "chest.png")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# =========================================================
# CHOOSE KERNEL
# =========================================================

# Gaussian
# kernel = get_gauss_kernel(2, 5)

# First derivative Gaussian
kernel = get_gauss_derivative_kernel(2, 5, "x")
kernel = get_gauss_derivative_kernel(2, 5, "y")

# LoG
# kernel = get_log_kernel(2, 5)

# Sobel
# kernel = get_sobel_kernel("x")
# kernel = get_sobel_kernel("y")


# =========================================================
# APPLY CONVOLUTION
# =========================================================

raw_output = convolve(image, kernel)


# =========================================================
# PREPARE FINAL OUTPUT
# =========================================================

# Gaussian is already positive and its kernel sums to 1
if np.all(kernel >= 0):

    output = np.clip(raw_output, 0, 255)
    output = output.astype(np.uint8)

# Derivative / LoG / Sobel
else:

    output = prepare_edge_output(raw_output)


# =========================================================
# DISPLAY
# =========================================================

# Create output directory if it doesn't exist
output_dir = os.path.join(script_dir, "Outputs", "Convolution")
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    output,
    cmap="gray",
    vmin=0,
    vmax=255
)
plt.title("Filtered")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "filtered_output.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# SAVE IMAGES
# =========================================================

cv2.imwrite(
    os.path.join(output_dir, "original.png"),
    image
)

cv2.imwrite(
    os.path.join(output_dir, "filtered_output.png"),
    output
)
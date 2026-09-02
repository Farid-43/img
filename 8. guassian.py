# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"img/retina_2.png", cv2.IMREAD_GRAYSCALE)

height = image.shape[0]
width = image.shape[1]


# kernel / filter from eqn
# gussian kernel

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
            output[i,j] = np.sum(kernel * padded_image[i:i+k_h,j:j+k_w])
            
    return output

def gussian_kernel(k_size, sigma):
    # formula of gussian
    # G(x,y) = 1/(2pi*sigma^2)*exp(-(x^2+y^2)/(2sigma^2))

    half = k_size // 2
    kernel = np.zeros((k_size, k_size))

    for i in range(k_size):
        for j in range(k_size):
            x = i - half
            y = j - half
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x)**2 + (y)**2) / (2 * sigma**2))
    
    kernel /= np.sum(kernel)
    return kernel

def first_derivative_gaussian(k_size, sigma):
    # formula of first derivative of gussian
    # G'(x, y) = (x * y / sigma**4) * exp(-(x^2+y^2)/(2sigma^2))

    half = k_size // 2
    kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            x = i - half
            y = j - half
            kernel[i, j] = (x * y / sigma**4) * np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel


box_kernel = np.ones((5, 5), dtype=np.float32) / 25

blur_image = conv2d(image, box_kernel)

guss_k = gussian_kernel(5, 1)
guassian_image = conv2d(image, guss_k)

guss_1dt = first_derivative_gaussian(5, 1)
guss_1dt_image = conv2d(image, guss_1dt)

plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(blur_image, cmap="gray", vmin=0, vmax=255)
plt.title("blur image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(guassian_image, cmap="gray", vmin=0, vmax=255)
plt.title("guassian")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(guss_1dt_image, cmap="gray", vmin=0, vmax=255)
plt.title("1d Gaussian")
plt.axis("off")

plt.tight_layout()
plt.savefig("out.png", dpi=300, bbox_inches="tight")
plt.show()

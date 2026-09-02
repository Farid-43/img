import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image using your format
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "img", "boat.jpg")
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

height, width = image_rgb.shape[:2]


# Exact 8-bit PDF and CDF from your repository
def get_pdf(channel):
    h, w = channel.shape[:2]
    pdf = np.zeros(256, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            pdf[channel[i, j]] += 1
    pdf /= h * w
    return pdf


def get_cdf(pdf):
    cdf = np.zeros_like(pdf)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf


# Direct 10-bit PDF for 1024 bins
def get_pdf_10bit(channel):
    h, w = channel.shape[:2]
    pdf = np.zeros(1024, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            pdf[channel[i, j]] += 1
    pdf /= h * w
    return pdf


# Manual equalization mapping directly to 1023
def equalize_to_10bit(channel):
    pdf = get_pdf(channel)
    cdf = get_cdf(pdf)

    T = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        T[i] = round(cdf[i] * 1023)

    new_channel = np.zeros((height, width), dtype=np.uint16)
    for i in range(height):
        for j in range(width):
            new_channel[i, j] = T[channel[i, j]]
    return new_channel


# --- 1. Original Histograms ---
r, g, b = cv2.split(image_rgb)
h, s, v = cv2.split(image_hsv)

pdf_r = get_pdf(r)
pdf_g = get_pdf(g)
pdf_b = get_pdf(b)
pdf_v = get_pdf(v)

# --- 2. RGB Equalization (10-bit) ---
new_r = equalize_to_10bit(r)
new_g = equalize_to_10bit(g)
new_b = equalize_to_10bit(b)
rgb_eq10 = cv2.merge([new_r, new_g, new_b])

# V channel for RGB-equalized image: V = max(R, G, B)
v_rgb_eq10 = np.maximum(np.maximum(new_r, new_g), new_b)

new_pdf_r = get_pdf_10bit(new_r)
new_pdf_g = get_pdf_10bit(new_g)
new_pdf_b = get_pdf_10bit(new_b)
pdf_v_rgb_eq10 = get_pdf_10bit(v_rgb_eq10)

# --- 3. HSV Equalization (10-bit) ---
v_eq10 = equalize_to_10bit(v)

# Convert back to RGB for display (scale 10-bit V to 8-bit, convert, then rescale to 1023)
v_8bit = np.round((v_eq10 / 1023.0) * 255.0).astype(np.uint8)
hsv_temp = cv2.merge([h, s, v_8bit])
hsv_rgb8 = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2RGB)

hsv_eq10_rgb = np.round((hsv_rgb8.astype(np.float32) / 255.0) * 1023.0).astype(
    np.uint16
)
r_hsv10, g_hsv10, b_hsv10 = cv2.split(hsv_eq10_rgb)

pdf_r_hsv10 = get_pdf_10bit(r_hsv10)
pdf_g_hsv10 = get_pdf_10bit(g_hsv10)
pdf_b_hsv10 = get_pdf_10bit(b_hsv10)
pdf_v_eq10 = get_pdf_10bit(v_eq10)

# --- 4. Plotting (Exact 3x3 layout from the PDF) ---
plt.figure(figsize=(14, 12))

# Row 1: Original 8-bit
plt.subplot(3, 3, 1)
plt.imshow(image_rgb)
plt.title("Original 8-bit")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.plot(pdf_r * (height * width), color="red")
plt.plot(pdf_g * (height * width), color="green")
plt.plot(pdf_b * (height * width), color="blue")
plt.title("Original 8-bit - RGB Histogram")
plt.xlim(0, 255)

plt.subplot(3, 3, 3)
plt.plot(pdf_v * (height * width))
plt.title("Original 8-bit - V Histogram")
plt.xlim(0, 255)

# Row 2: RGB Equalized (10-bit)
plt.subplot(3, 3, 4)
plt.imshow(rgb_eq10.astype(np.float32) / 1023.0)
plt.title("RGB Equalized (10-bit)")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.plot(new_pdf_r * (height * width), color="red")
plt.plot(new_pdf_g * (height * width), color="green")
plt.plot(new_pdf_b * (height * width), color="blue")
plt.title("RGB Equalized (10-bit) - RGB Histogram")
plt.xlim(0, 1023)

plt.subplot(3, 3, 6)
plt.plot(pdf_v_rgb_eq10 * (height * width))
plt.title("RGB Equalized (10-bit) - V Histogram")
plt.xlim(0, 1023)

# Row 3: HSV Equalized (10-bit)
plt.subplot(3, 3, 7)
plt.imshow(hsv_eq10_rgb.astype(np.float32) / 1023.0)
plt.title("HSV Equalized (10-bit)")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.plot(pdf_r_hsv10 * (height * width), color="red")
plt.plot(pdf_g_hsv10 * (height * width), color="green")
plt.plot(pdf_b_hsv10 * (height * width), color="blue")
plt.title("HSV Equalized (10-bit) - RGB Histogram")
plt.xlim(0, 1023)

plt.subplot(3, 3, 9)
plt.plot(pdf_v_eq10 * (height * width))
plt.title("HSV Equalized (10-bit) - V Histogram")
plt.xlim(0, 1023)

plt.tight_layout()
plt.savefig("result.png", dpi=300, bbox_inches="tight")
plt.show()
# 5. hist qualization rgb.py
# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image loading
image = cv2.imread(r"img/boat.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

image = image_rgb

print(image.shape) # (height, width, 3)

height = image.shape[0]
width = image.shape[1]

def get_pdf(channel):
    h, w = channel.shape[:2]
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
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

r, g, b = cv2.split(image)

# r = image[:, :, 0]
# g = image[:, :, 1]
# b = image[:, :, 2]

# pdf & cdf
pdf_r = get_pdf(r)
pdf_g = get_pdf(g)
pdf_b = get_pdf(b)
cdf_r = get_cdf(pdf_r)
cdf_g = get_cdf(pdf_g)
cdf_b = get_cdf(pdf_b)

# hist mapping
hist_eq_r = np.zeros_like(pdf_r)
hist_eq_g = np.zeros_like(pdf_g)
hist_eq_b = np.zeros_like(pdf_b)
for i in range(256):
    hist_eq_r[i] = round(cdf_r[i] * 255)
    hist_eq_g[i] = round(cdf_g[i] * 255)
    hist_eq_b[i] = round(cdf_b[i] * 255)

new_r = np.zeros_like(r)
new_g = np.zeros_like(g)
new_b = np.zeros_like(b)
for i in range(height):
    for j in range(width):
        new_r[i,j] = hist_eq_r[r[i,j]]
        new_g[i,j] = hist_eq_g[g[i,j]]
        new_b[i,j] = hist_eq_b[b[i,j]]

# new pdf & cdf
new_pdf_r = get_pdf(new_r)
new_pdf_g = get_pdf(new_g)
new_pdf_b = get_pdf(new_b)
new_cdf_r = get_cdf(new_pdf_r)
new_cdf_g = get_cdf(new_pdf_g)
new_cdf_b = get_cdf(new_pdf_b)

new_image = image.copy()
new_image[:, :, 0] = new_r
new_image[:, :, 1] = new_g
new_image[:, :, 2] = new_b

# plotting (orginal vs log)
plt.figure(figsize=(8, 16)) # (width, height)

plt.subplot(3, 2, 1) # (rows, columns, index)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(new_image, cmap="gray")
plt.title("eqalized")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.plot(pdf_r, color="red")
plt.plot(pdf_g, color="green")
plt.plot(pdf_b, color="blue")
plt.title("PDF")
plt.ylabel("probability")
plt.legend(["Original R", "Original G", "Original B"])

plt.subplot(3, 2, 4)
plt.plot(cdf_r, color="red")
plt.plot(cdf_g, color="green")
plt.plot(cdf_b, color="blue")

plt.title("CDF")
plt.ylabel("probability")
plt.legend(["Original R", "Original G", "Original B"])

plt.subplot(3, 2, 5)
plt.plot(new_pdf_r, color="red")
plt.plot(new_pdf_g, color="green")
plt.plot(new_pdf_b, color="blue")
plt.title("PDF")
plt.ylabel("probability")
plt.legend(["Equalized R", "Equalized G", "Equalized B"])

plt.subplot(3, 2, 6)
plt.plot(new_cdf_r, color="red")
plt.plot(new_cdf_g, color="green")
plt.plot(new_cdf_b, color="blue")
plt.title("CDF")
plt.ylabel("probability")
plt.legend(["Equalized R", "Equalized G", "Equalized B"])

# plt.tight_layout()
plt.savefig(r"out.png", dpi=300, bbox_inches="tight")
plt.show()


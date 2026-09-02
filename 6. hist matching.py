# 6. hist matching.py
# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image loading
image = cv2.imread(r"img/boat.jpg")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

ref = cv2.imread(r"img/power_plant.jpg")
ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)

image = image_hsv

print(image.shape) # (height, width, 3)

height = image.shape[0]
width = image.shape[1]

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
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

h, _, _ = cv2.split(image)

# r = image[:, :, 0]
# g = image[:, :, 1]
# b = image[:, :, 2]

# pdf & cdf (image)
pdf_h = get_pdf(h)
cdf_h = get_cdf(pdf_h)


ref_h, _, _ = cv2.split(ref_hsv)

# pdf & cdf (ref)
pdf_ref_h = get_pdf(ref_h)
cdf_ref_h = get_cdf(pdf_ref_h)

new_h = np.zeros_like(h)
for i in range(height):
    for j in range(width):
        r = h[i,j]
        cdf_val = cdf_h[r]
        
        # find the closest cdf value in ref image
        diff_array = np.abs(cdf_ref_h - cdf_val)
        s = np.argmin(diff_array)
        new_h[i,j] = s
        
new_image = image.copy()
new_image[:, :, 0] = new_h

new_pdf_h = get_pdf(new_h)
new_cdf_h = get_cdf(new_pdf_h)

# plotting (orginal vs log)
plt.figure(figsize=(8, 16)) # (width, height)

plt.subplot(3, 2, 1) # (rows, columns, index)
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(3, 2, 2)
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
plt.imshow(new_image, cmap="gray")
plt.title("eqalized")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.plot(pdf_h, color="red")
plt.plot(new_pdf_h, color="green")
plt.title("PDF")
plt.ylabel("probability")
plt.legend(["Original H", "new H"])

plt.subplot(3, 2, 4)
plt.plot(cdf_h, color="red")
plt.plot(cdf_ref_h, color="green")
plt.title("CDF")
plt.ylabel("probability")
plt.legend(["Original H", "new H"])

plt.subplot(3, 2, 5)
plt.plot(pdf_ref_h, color="red")
plt.title("PDF")
plt.ylabel("probability")
plt.legend(["Ref H"])

plt.subplot(3, 2, 6)
plt.plot(cdf_ref_h, color="red")
plt.title("CDF")
plt.ylabel("probability")
plt.legend(["Ref H"])

# plt.tight_layout()
plt.savefig(r"out.png", dpi=300, bbox_inches="tight")
plt.show()


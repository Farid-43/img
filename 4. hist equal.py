# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image loading
image = cv2.imread(r"img/retina_2.png", cv2.IMREAD_GRAYSCALE)
# other options: 
# boat_bgr = cv2.imread(r'task/img/boat.jpg')

height = image.shape[0]
width = image.shape[1]

def get_pdf(image):
    pdf = np.zeros(256, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            pdf[image[i,j]] += 1
    pdf /= (height * width)
    return pdf

def get_cdf(pdf):
    cdf = np.zeros_like(pdf)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

# pdf & cdf
pdf = get_pdf(image)
cdf = get_cdf(pdf)

# hist mapping
T = np.zeros_like(pdf)
for i in range(256):
    T[i] = round(cdf[i] * 255)

new_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        new_image[i,j] = T[image[i,j]]

# new pdf & cdf
new_pdf = get_pdf(new_image)
new_cdf = get_cdf(new_pdf)

# plotting (orginal vs log)
plt.figure(figsize=(8, 15)) # (width, height)

plt.subplot(3, 2, 1) # (rows, columns, index)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(new_image, cmap="gray")
plt.title("eqalized")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.plot(pdf, color="black")
plt.title("PDF")
plt.xlabel("intensity")
plt.ylabel("probability")

plt.subplot(3, 2, 4)
plt.plot(cdf, color="black")
plt.title("CDF")
plt.xlabel("intensity")
plt.ylabel("probability")

plt.subplot(3, 2, 5)
plt.plot(new_pdf, color="black")
plt.title("New PDF")
plt.xlabel("intensity")
plt.ylabel("probability")

plt.subplot(3, 2, 6)
plt.plot(new_cdf, color="black")
plt.title("New CDF")
plt.xlabel("intensity")
plt.ylabel("probability")

plt.tight_layout()
plt.savefig(r"out.png", dpi=300, bbox_inches="tight")
plt.show()


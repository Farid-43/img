# 11. butterworth notch.py
# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt

# take input
img_input = cv2.imread(r'img/pnois2.jpg', 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img) # it gives the fourier transform of the image
ft_shift = np.fft.fftshift(ft) # it shifts DC component to center

# get magnitude and phase
magnitude_spectrum_ac = np.abs(ft_shift) #actual magnitude spectrum
ang = np.angle(ft_shift)

#Just for visualization
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

# phase visualization if needed
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 


# Apply filter here: 
# Design H(u,v) an ideal notch reject filter to remove periodic noise and just do 
# F(u,v) = F(u,v)*H(u,v) to get the filtered result here F(u,v) is the  magnitude spectrum 
points = [
    (261,261),
    (251,251)
]

D0 = 5 
n = 2
h, w = magnitude_spectrum_ac.shape

H = np.ones_like(magnitude_spectrum_ac)

# apply Butterworth
for i in range(h):
    for j in range(w):
        for (x, y) in points:
            D = ((i-x)**2 + (j-y)**2)**0.5
            if D == 0:
                H[i, j] = 0.0
            else:
                H[i, j] *= 1 / (1 + (D0 / D)**(2 * n))

magnitude_spectrum_ac = magnitude_spectrum_ac * H

## phase add F(u,v)=∣F(u,v)∣*e^jθ(u,v) combine magnitude and phase to get the final result
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

final_result_vis = 20 * np.log(np.abs(final_result)+1)
final_result_vis = cv2.normalize(final_result_vis, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

# inverse fourier transform and get the image back
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)


## plot

# --- Plotting Results ---
plt.figure(figsize=(12, 12))

plt.subplot(2, 3, 1)
plt.imshow(img_input, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray", vmin=0, vmax=255)
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(ang_, cmap="gray", vmin=0, vmax=255)
plt.title("Phase")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img_back_scaled, cmap="gray", vmin=0, vmax=255)
plt.title("Inverse transform")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(final_result_vis, cmap="gray", vmin=0, vmax=255)
plt.title("Final Result Vis")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(H, cmap="gray", vmin=0, vmax=1)
plt.title("Filter")
plt.axis("off")

plt.tight_layout()
plt.savefig("out.png", dpi=300, bbox_inches="tight")
plt.show()

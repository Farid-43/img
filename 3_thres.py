# 3_thres.py
# import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image loading
image = cv2.imread(r"img/chest.png", cv2.IMREAD_GRAYSCALE)
# other options: 
# boat_bgr = cv2.imread(r'task/img/boat.jpg')


height = image.shape[0]
width = image.shape[1]
thres1 = 50
thres2 = 150

output = np.empty_like(image)

# apply log 
for i in range(height):
    for j in range(width):
        r = image[i, j].astype(np.float32)
        
        if r < thres1:
            s = 0
        elif r < thres2:
            s = 100
        else:
            s = 255
        
        output[i, j] = np.uint8(s)

# plotting (orginal vs log)
plt.figure(figsize=(8, 6)) # (width, height)

plt.subplot(1, 2, 1) # (rows, columns, index)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("Thres")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"out.png", dpi=300, bbox_inches="tight")
plt.show()


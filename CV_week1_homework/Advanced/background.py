import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


img_kb = cv2.imread('kobe_bg.jpg', 1)
image_show(img_kb)
print(img_kb.shape)

# BGR to HSV
hsv = cv2.cvtColor(img_kb, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 70, 70])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Erode & dilute
erode = cv2.erode(mask, None, iterations=1)
dilate = cv2.dilate(erode, None, iterations=1)

# HSV to BGR
for i in range(img_kb.shape[0]):
    for j in range(img_kb.shape[1]):
        if dilate[i, j] == 255:
            img_kb[i, j] = (0, 0, 255)

image_show(img_kb)

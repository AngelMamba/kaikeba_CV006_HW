import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_show(img, size=(2, 2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


img_dark = cv2.imread('dark.jpg', 1)
my_show(img_dark, size=(6, 6))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)


img_brighter = adjust_gamma(img_dark, 2)
my_show(img_brighter, size=(6, 6))

# Histogram Equalization in BGR
plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color='r')
plt.subplot(122)
plt.hist(img_brighter.flatten(), 256, [0, 256], color='b')
plt.show()

# Histogram Equalization in YUV
img_yuv = cv2.cvtColor(img_dark, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

my_show(img_output, size=(8, 8))

plt.subplot(131)
plt.hist(img_dark.flatten(), 256, [0, 256], color='r')
plt.subplot(132)
plt.hist(img_brighter.flatten(), 256, [0, 256], color='b')
plt.subplot(133)
plt.hist(img_output.flatten(), 256, [0, 256], color='g')
plt.show()

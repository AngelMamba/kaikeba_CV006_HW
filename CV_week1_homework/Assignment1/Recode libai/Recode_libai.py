import cv2
import matplotlib.pyplot as plt


img_libai = cv2.imread('libai.png', 0)
plt.figure(figsize=(10, 8))
plt.imshow(img_libai, cmap='gray')
plt.show()

img_erode = cv2.erode(img_libai, None, iterations=1)
plt.figure(figsize=(10, 8))
plt.imshow(img_erode, cmap='gray')
plt.show()

img_dilate = cv2.dilate(img_libai, None, iterations=1)
plt.figure(figsize=(10, 8))
plt.imshow(img_dilate, cmap='gray')
plt.show()

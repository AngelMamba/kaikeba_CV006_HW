import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# Show original image
img_lenna = cv2.imread('lenna.jpg', 1)
image_show(img_lenna)
print(img_lenna.shape)

# Covert BGR to YUV
img_yuv = cv2.cvtColor(img_lenna, cv2.COLOR_BGR2YUV)
print('YUV = ', img_yuv)

# Change U and V (Y remains same)
du = 5
dv = 12
img_yuv2 = img_yuv + [0, (du-10)*10.0, (dv-10)*10.0]
img_yuv2 = np.array(img_yuv2, np.uint8)

# Return to RGB channel
img_new = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2RGB)
plt.imshow(img_new)
plt.show()


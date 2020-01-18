import cv2
import matplotlib.pyplot as plt

img_origin = cv2.imread('lenna.jpg', 1)
print(img_origin.shape)

# cv2.imshow('lenna', img_origin)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

# plt.imshow(img_origin)
# plt.show()

# plt.figure(figsize=(2, 2)) # change size. Default size = [8,6]
# plt.rcParams['figure.figsize'] = [4, 4]

# plt.subplot(121) # subplot
# plt.imshow(img_origin)
# plt.subplot(122)
plt.imshow(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
plt.show()


def my_show(img, size=(2,2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# my_show(img_origin)

# Image Crop
img_crop = img_origin[150:200, 50:100]
# plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
# plt.show()

# Gray
plt.imshow(img_origin, cmap='gray')
# plt.show()

# Split
B, G, R = cv2.split(img_origin)
# cv2.imshow('B', B)
# cv2.imshow('G', G)
# cv2.imshow('R', R)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

plt.subplot(311)
plt.imshow(B, cmap='gray')
plt.subplot(312)
plt.imshow(G, cmap='gray')
plt.subplot(313)
plt.imshow(R, cmap='gray')
plt.show()


def img_cooler(img, b_increase, r_decrease):
    B,G,R = cv2.split(img)
    b_lim = 255 - b_increase
    B[B > b_lim] = 255
    B[B <= b_lim] = (b_increase + B[B <= b_lim]).astype(img.dtype)
    r_lim = r_decrease
    R[R < r_lim] = 0
    R[R >= r_lim] = (R[R >= r_lim] - r_decrease).astype(img.dtype)
    return cv2.merge((B,G,R))


img_cool = img_cooler(img_origin, 20, 10)
my_show(img_cool)
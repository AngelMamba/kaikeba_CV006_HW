import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_show(img, size=(2, 2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


img_kobe = cv2.imread('kobe.jpg', 1)
my_show(img_kobe, size=(8, 8))

pts1 = np.float32([[0, 0], [0, 350], [500, 0], [500, 350]])
pts2 = np.float32([[5, 10], [10, 300], [400, 10], [400, 200]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img_kobe, M, (500, 400))
my_show(img_warp)
print('M=')
print(M)

# Rotation
M_r1 = cv2.getRotationMatrix2D((img_kobe.shape[1] / 2, img_kobe.shape[0] / 2), 30, 1) # center, angle, scale
img_rotate = cv2.warpAffine(img_kobe, M_r1, (img_kobe.shape[1], img_kobe.shape[0]))
my_show(img_rotate, size=[8, 8])
print('M_r1=', M_r1)
print(M_r1)

M_r2 = cv2.getRotationMatrix2D((img_kobe.shape[1] / 2, img_kobe.shape[0] / 3), 20, 0.3) # center, angle, scale
img_rotate2 = cv2.warpAffine(img_kobe, M_r2, (img_kobe.shape[1], img_kobe.shape[0]))
my_show(img_rotate2, size=[8, 8])
print('M_r2=')
print(M_r2)

# Affine Transform
# print(img_kobe.shape)
rows, cols, ch = img_kobe.shape
pts_aff_1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts_aff_2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M_aff = cv2.getAffineTransform(pts_aff_1, pts_aff_2)
img_affine = cv2.warpAffine(img_kobe, M_aff, (cols, rows))
my_show(img_affine, size=[8, 8])
print('M_aff=')
print(M_aff)


# Perspective Transform
def random_warp(img, row, col):
    height, width, channels = img_kobe.shape
    random_margin = 60
    x1 = np.random.randint(-random_margin, random_margin)
    y1 = np.random.randint(-random_margin, random_margin)
    x2 = np.random.randint(width - random_margin - 1, width - 1)
    y2 = np.random.randint(-random_margin, random_margin)
    x3 = np.random.randint(width - random_margin - 1, width - 1)
    y3 = np.random.randint(height - random_margin - 1, height - 1)
    x4 = np.random.randint(-random_margin, random_margin)
    y4 = np.random.randint(height - random_margin - 1, height - 1)

    dx1 = np.random.randint(-random_margin, random_margin)
    dy1 = np.random.randint(-random_margin, random_margin)
    dx2 = np.random.randint(width - random_margin - 1, width - 1)
    dy2 = np.random.randint(-random_margin, random_margin)
    dx3 = np.random.randint(width - random_margin - 1, width - 1)
    dy3 = np.random.randint(height - random_margin - 1, height - 1)
    dx4 = np.random.randint(-random_margin, random_margin)
    dy4 = np.random.randint(height - random_margin - 1, height - 1)

    pts1_random = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2_random = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp_random = cv2.getPerspectiveTransform(pts1_random, pts2_random)
    img_warp_random = cv2.warpPerspective(img, M_warp_random, (width, height))
    return M_warp_random, img_warp_random


M_warp_random, kobe_warp_random = random_warp(img_kobe, img_kobe.shape[0], img_kobe.shape[1])
my_show(kobe_warp_random, size=[8, 8])
print('M_warp_random=')
print(M_warp_random)
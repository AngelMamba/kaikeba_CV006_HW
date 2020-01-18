import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_show(img, size=(8, 8)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def image_crop(img, height, width):
    # if height.size != 2 or height[0] >= height[1]:
    #     print('Wrong dimension of height!')
    # if width.size != 2 or width[0] >= width[1]:
    #     print('Wrong dimension of width!')
    cropped_img = img[height[0]:height[1], width[0]:width[1]]
    return cropped_img


def color_shift(img, shift_bgr):
    # shift_bgr: B, G, R
    B = 0
    G = 1
    R = 2
    # if shift_bgr.size != 3 :
    #     print('Wrong dimension of shift_bgr!')
    b_origin, g_origin, r_origin = cv2.split(img)
    b_shift = (b_origin + shift_bgr[B]).astype(img.dtype)
    g_shift = (g_origin + shift_bgr[G]).astype(img.dtype)
    r_shift = (r_origin + shift_bgr[R]).astype(img.dtype)

    # Check constrains
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b_shift[i, j] = b_shift[i, j] if b_shift[i, j] >= 0 else 0
            g_shift[i, j] = g_shift[i, j] if g_shift[i, j] >= 0 else 0
            r_shift[i, j] = r_shift[i, j] if r_shift[i, j] >= 0 else 0

    print('b_shift = ', b_shift)
    return cv2.merge((b_shift, g_shift, r_shift))


def image_rotate(img, rot_center, rot_angle, rot_scale):
    # rot_angle (deg)

    if rot_center[0] > img.shape[0] or rot_center[1] > img.shape[1]:
        print('Out of image shape!')

    rot_M = cv2.getRotationMatrix2D(rot_center, rot_angle, rot_scale)
    rotated_image = cv2.warpAffine(img, rot_M, (img.shape[1], img.shape[0]))
    print('rot_M = ', rot_M)
    return rotated_image


def perspective_transform(img, target_point):
    source_point = np.float32([[0, 0], [img.shape[0], 0], [img.shape[0], img.shape[1]], [0, img.shape[1]]])
    PersTrans_M = cv2.getPerspectiveTransform(source_point, target_point)
    print('PersTrans_M = ', PersTrans_M)
    return cv2.warpPerspective(img, PersTrans_M, (img.shape[0], img.shape[1]))


img_lenna = cv2.imread('lenna.jpg', 1)
image_show(img_lenna)
print(img_lenna.shape)

cropped_img = image_crop(img_lenna, (100, 150), (100, 150))
plt.subplot(221)
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
# image_show(cropped_img)

color_shift_img = color_shift(img_lenna, (20, 10, -5))
plt.subplot(222)
plt.imshow(cv2.cvtColor(color_shift_img, cv2.COLOR_BGR2RGB))
# image_show(color_shift_img)

rotated_img = image_rotate(img_lenna, (200, 100), 20, 1)
plt.subplot(223)
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
# image_show(rotated_img)

target_point = np.float32([[100, 10], [400, 10], [490, 490], [10, 490]])
perspective_transformed_img = perspective_transform(img_lenna, target_point)
plt.subplot(224)
plt.imshow(cv2.cvtColor(perspective_transformed_img, cv2.COLOR_BGR2RGB))
# image_show(perspective_transformed_img)
plt.show()

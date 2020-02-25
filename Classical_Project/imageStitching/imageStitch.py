import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def detect_KP_DES(img):
    """

    :param img:
    :return KP:
    :return DES:
    """
    sift = cv2.xfeatures2d.SIFT_create()
    KP = sift.detect(img)
    KP, DES = sift.compute(img, KP)
    return KP, DES


def img_stitching(image1, image2):
    # get key points from img1 & img2 based on SIFT
    KP1, DES1 = detect_KP_DES(image1)
    KP2, DES2 = detect_KP_DES(image2)

    # Find the match points from img1 & img2
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(DES1, DES2)

    # Set threshold=0.8 and choose good match points
    goodMatch = []
    for m in range(0, len(matches) - 1):
        if matches[m].distance < 0.8 * matches[m+1].distance:
            goodMatch.append(matches[m])

    # draw the good match points between img1 & img2
    img_goodmatch = cv2.drawMatches(image1, KP1, image2, KP2, goodMatch, flags=2, outImg=None)

    # Find the Homography Matrix between img1 & img2
    pts1 = np.float32([KP1[p.queryIdx].pt for p in goodMatch])
    pts2 = np.float32([KP2[p.trainIdx].pt for p in goodMatch])
    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # Make img2 do perspective transformation and add img2 to img1
    #     shift = np.array([[1.0,0,w],[0,1.0,0],[0,0,1.0]])
    #     M = np.dot(shift,H)
    img_out = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    img_out[0:image1.shape[0], 0:image1.shape[1]] = image1

    return img_goodmatch, img_out


if __name__ == "__main__":
    img1 = cv2.imread('img1.png', 1)
    img2 = cv2.imread('img2.png', 1)
    show_image(img1)
    show_image(img2)
    print('img1 size is ', img1.shape)
    print('img2 size is ', img2.shape)

    img_goodmatch, img_stitched = img_stitching(img1, img2)
    show_image(img_goodmatch)
    show_image(img_stitched)

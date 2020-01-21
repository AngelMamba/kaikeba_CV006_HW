import cv2
import numpy as np
import matplotlib.pyplot as plt


# Show original image
img_lenna_sp = cv2.imread('lenna_sp.jpg', 1)
plt.imshow(cv2.cvtColor(img_lenna_sp, cv2.COLOR_BGR2RGB))
plt.show()
print(img_lenna_sp.shape)


def medianBlur(img, kernel, padding_way):

    # Define median blur window
    N = kernel[0]
    M = kernel[1]
    window = np.zeros((N, M), dtype=int)

    # Edge area to be handled
    edge_h = int(0+(N-1)/2)
    edge_w = int(0+(M-1)/2)

    # Image data expansion
    if padding_way == 'REPLICA':
        r1 = img[0, :, :]  # read first row
        r1 = r1[None]  # dim 2 --> 3
        r1 = np.repeat(r1, edge_h, axis=0)  # repeat to fill edge_h
        rn = img[-1, :, :]  # read last row
        rn = rn[None]  # dim 2 --> 3
        rn = np.repeat(rn, edge_h, axis=0)  # repeat to fill edge_h
        img = np.append(r1, img, axis=0)  # concatenate r1 and img
        img = np.append(img, rn, axis=0)  # concatenate img and rn
        c1 = img[:, 0, :]  # read first column
        c1 = c1[:, None]  # dim 2 --> 3
        c1 = np.repeat(c1, edge_w, axis=1)  # repeat to fill edge_h
        cn = img[:, -1, :]  # read first column
        cn = cn[:, None]  # dim 2 --> 3
        cn = np.repeat(cn, edge_w, axis=1)  # repeat to fill edge_h
        img = np.append(c1, img, axis=1)  # concatenate r1 and img
        img = np.append(img, cn, axis=1)  # concatenate img and rn

    elif padding_way == 'ZERO':
        r = np.zeros((edge_h, int(img.shape[1]), int(img.shape[2])), dtype=int)  # create row zeros
        img = np.append(r, img, axis=0)  # concatenate r and img
        img = np.append(img, r, axis=0)  # concatenate img and r

        c = np.zeros((int(img.shape[0]), edge_w, int(img.shape[2])), dtype=int)  # create column zeros
        img = np.append(c, img, axis=1)  # concatenate c and img
        img = np.append(img, c, axis=1)  # concatenate img and c

    else:
        print('Wrong padding way!')
        return False

    # Median filter
    range_h1 = (0+(N-1)/2).astype(int)
    range_h2 = (img.shape[0]-(N-1)/2).astype(int)
    range_w1 = (0+(M-1)/2).astype(int)
    range_w2 = (img.shape[1]-(M-1)/2).astype(int)
    for d in range(img.shape[2]):
        for h in range(range_h1, range_h2):
            for w in range(range_w1, range_w2):
                window_h1 = (h-(N-1)/2).astype(int)
                window_h2 = (h+(N-1)/2+1).astype(int)
                window_w1 = (w-(M-1)/2).astype(int)
                window_w2 = (w+(M-1)/2+1).astype(int)
                window = img[window_h1:window_h2, window_w1:window_w2, d]

                median = np.sort(window, axis=None, kind='quicksort')
                median_n = ((N*M-1)/2).astype(int)
                medianValue = median[median_n]
                img[h, w, d] = medianValue

        img_medianBlur = np.array(img[range_h1:range_h2, range_w1:range_w2, :], dtype='uint8')
    return img_medianBlur


window_NM = np.array([9, 9], dtype=int)
img_medianBlur = medianBlur(img_lenna_sp, window_NM, 'ZERO')
plt.imshow(cv2.cvtColor(img_medianBlur, cv2.COLOR_BGR2RGB))
plt.show()

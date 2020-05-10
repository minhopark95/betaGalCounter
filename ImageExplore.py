import cv2
import numpy as np
from os import listdir
from scipy.stats import norm
import statistics as stats
from matplotlib import pyplot as plt
from matplotlib import mlab

SIGMA_MULTIPLIER = 4
G_RES_MULTIPLER = 0.2
SIG_THRESH_MULT = 0.3


def gradient_correction(img):
    # defining variables
    mu, sigma = norm.fit(img.ravel())

    # Shrink and apply median blur to make a gradient image (g_img)
    g_img = cv2.resize(img, None, fx=G_RES_MULTIPLER, fy=G_RES_MULTIPLER, interpolation=cv2.INTER_NEAREST)
    g_img = cv2.medianBlur(g_img, 7)
    g_img = cv2.subtract(g_img, 4*sigma)

    print("shrunk size:", g_img.shape)
    g_img = cv2.resize(src = g_img, dsize = (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    print("reexpaneded size:", g_img.shape)

    img = cv2.subtract(img, g_img)
    # only care about the first two values, height and width
    print('in gradient correction: resized image')

    cv2.imshow('subtracted image', img)

    mode = stats.mode(img.ravel())
    mu, sigma = norm.fit(img.ravel())

    threshLevel = mode - 0.5*sigma
    ___, thresh_img = cv2.threshold(img, threshLevel, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresholded image', thresh_img)
    # n, bins, patches = plt.hist(img.ravel(), 256, [0, 255], normed=True)
    # normalCurve = norm.pdf(bins, mode, sigma)
    # plt.plot(bins, normalCurve, 'r--', linewidth=2)
    # plt.show()
    # make the correction map, a map of each pixel's deviation from the average
#    for col in range(height):
#        c_map.append([])
#        for row in range(width):
#            c_map[col].append(c_value - g_img[col][row])

#    big_height, big_width = img.shape[:2]

    # apply the correction map to the corresponding pixel in the full sized image
#    for col in range(big_height):
#        for row in range(big_width):
#            c_col = int(col * g_res)
#            c_row = int(row * g_res)
#            sig_thresh = g_img[c_col][c_row] * SIG_THRESH_MULT
#            if img[col][row] > sig_thresh:  # reverse the < to undo comments
#                img[col][row] = 0
#            else:  # Set colonies to 0 for circle detect
#                img[col][row] = 255

#    del g_img
#    del c_map

    return img


def cellFinder(file: chr):
    img = cv2.imread('./Test Images/' + file)
    r, g, b = cv2.split(img)
    print(g.shape)
    g = gradient_correction(g)
    # cv2.imshow('gradient corrected image', g)
    # cv2.waitKey()


def bGalFinder(file: chr):
    img = cv2.imread('./Test Images/' + file)
    r, g, b = cv2.split(img)

    bDiff = cv2.subtract(r, b)

    mu, sigma = norm.fit(bDiff.ravel())
    thresh_val = mu + SIGMA_MULTIPLIER * sigma
    __, bDiffThresh = cv2.threshold(bDiff, thresh_val, 123, cv2.THRESH_BINARY)

    # Plot out with matplotlib to get a pixel location
    # cv2.imshow('Blue Thresholded', bDiffThresh)
    cv2.imshow('original image', img)
    kernErode = np.ones((2, 2), np.uint8)
    kernDilate = np.ones((3, 3), np.uint8)
    bDiffErode = cv2.erode(bDiffThresh, kernErode)
    bDiffDialated = cv2.dilate(bDiffErode, kernDilate)

    # cv2.imshow('dilated image', bDiffDialated)
    edges = cv2.Canny(bDiffDialated, 175, 200)
    cv2.imshow('Canny Edges', edges)

    ____, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursImg = cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    cv2.imshow('contoured Images', contoursImg)
    # plt.figure(1)
    # plt.imshow(bDiff)
    # plt.show()
    # test = cv2.add(b, bDiffThresh)

    # mu, sigma = norm.fit(bDiff.ravel())
    # n, bins, patches = plt.hist(bDiff.ravel(), 256, [0, 50], normed=True)
    # normalCurve = norm.pdf(bins, mu, sigma)
    # plt.plot(bins, normalCurve, 'r--', linewidth=2)
    # plt.show()
    cv2.waitKey()


def main():
    print(cv2.__version__)
    files = listdir('./Test Images')
    for f in files:
        # bGalFinder(file=f)
        cellFinder(file=f)


if __name__ == "__main__":
    main()

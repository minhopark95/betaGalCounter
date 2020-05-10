import cv2
import numpy as np
from os import listdir
from scipy.stats import norm
import statistics as stats
from matplotlib import pyplot as plt
from matplotlib import mlab

BLUE_SIGMA_MULTIPLIER = 4
CELL_SIGMA_MULTILIER = 4
G_RES_MULTIPLER = 0.2
CONTRAST_MULTIPLIER = 0.75
CLEANUP_LOOPS = 2


def increaseContrast(img):
    # get stats real quick
    mu, sigma = norm.fit(img.ravel())

    # Shrink and apply median blur and expand to make a gradient image (g_img)
    g_img = cv2.resize(img, None, fx=G_RES_MULTIPLER, fy=G_RES_MULTIPLER, interpolation=cv2.INTER_NEAREST)
    g_img = cv2.medianBlur(g_img, 7)
    g_img = cv2.subtract(g_img, CELL_SIGMA_MULTILIER * sigma)
    g_img = cv2.resize(src=g_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

    # subtract the gradient image from the data channel
    img = cv2.subtract(img, g_img)
    return img


def identifyCells(img):
    # get centering statistics to identify the cells
    mode = stats.mode(img.ravel())
    mu, sigma = norm.fit(img.ravel())

    # threshold image
    threshLevel = mode - CONTRAST_MULTIPLIER * sigma
    ___, thresh_img = cv2.threshold(img, threshLevel, 255, cv2.THRESH_BINARY)

    # do an erode followed by dialation x2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = thresh_img
    for i in range(CLEANUP_LOOPS):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    # Invert the image and find contours
    closed = cv2.bitwise_not(closed)
    ____, contours, _____ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Go smoothen cell exteriors with convex hull and drop any small cells
    hull = []
    for i in range(len(contours)):
        size = cv2.contourArea(contours[i])
        if size > 10:
            hull.append(cv2.convexHull(contours[i], False))

    # draw the convex hulls
    cv2.drawContours(closed, hull, -1, 123, 2)

    cv2.imshow('thresholded image', closed)
    cv2.imshow('original image', img)

    return img


def cellFinder(file: chr):
    img = cv2.imread('./Test Images/' + file)
    r, g, b = cv2.split(img)

    r = increaseContrast(img=r)
    r = identifyCells(img=r)
    cv2.waitKey()


def bGalFinder(file: chr):
    img = cv2.imread('./Test Images/' + file)
    r, g, b = cv2.split(img)

    bDiff = cv2.subtract(r, b)

    mu, sigma = norm.fit(bDiff.ravel())
    thresh_val = mu + BLUE_SIGMA_MULTIPLIER * sigma
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

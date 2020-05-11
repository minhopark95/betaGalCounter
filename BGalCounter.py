import cv2
import numpy as np
from scipy.stats import norm
import statistics as stats
from matplotlib import pyplot as plt
import os
import sys
import argparse

# # Global variables to count Cells
CELL_SD_THRESHOLD: float
G_RES_MULTIPLIER: float
CONTRAST_MULTIPLIER: int
CLEANUP_LOOPS: int
MIN_CELL_SIZE: int
#
# # Global variables to count Beta Gal Cells
BLUE_SD_THRESHOLD: int
MIN_BGAL_SIZE: int
#
# # Global variable to show Testing Plots
TESTING: bool


def increaseContrast(img):
    # get stats real quick
    mu, sigma = norm.fit(img.ravel())

    # Shrink and apply median blur and expand to make a gradient image (g_img)
    g_img = cv2.resize(img, None, fx=G_RES_MULTIPLIER, fy=G_RES_MULTIPLIER, interpolation=cv2.INTER_NEAREST)
    g_img = cv2.medianBlur(g_img, 7)
    g_img = cv2.subtract(g_img, CONTRAST_MULTIPLIER * sigma)
    g_img = cv2.resize(src=g_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

    # subtract the gradient image from the data channel
    img = cv2.subtract(img, g_img)

    del mu, sigma, g_img
    return img


def identifyCells(img):
    # get centering statistics to identify the cells
    mode = stats.mode(img.ravel())
    mu, sigma = norm.fit(img.ravel())

    # threshold image
    threshLevel = mode - CELL_SD_THRESHOLD * sigma
    ___, thresh_img = cv2.threshold(img, threshLevel, 255, cv2.THRESH_BINARY)

    # do an erode followed by dialation x2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(CLEANUP_LOOPS):
        closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    # Invert the image and find contours
    closed = cv2.bitwise_not(closed)
    ____, contours, ____ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Go smoothen cell exteriors with convex hull and drop any small cells
    hull = []
    for i in range(len(contours)):
        size = cv2.contourArea(contours[i])
        if size > MIN_CELL_SIZE:
            hull.append(cv2.convexHull(contours[i], False))

    # draw the convex hulls
    cv2.drawContours(closed, hull, -1, 123, 2)

    # outputs
    if TESTING:
        # plot out
        cv2.imshow("Contrast Image", img)
        cv2.drawContours(img, hull, -1, 123, 1)
        cv2.imshow("Contours", img)
        areas = []
        for i in range(len(hull)):
            areas.append(cv2.contourArea(hull[i]))
        plt.hist(areas, 30, [0, 500])
        plt.title('All filtered identified cells')
        plt.show()

    # Make output mask and return it along with the number of cells
    outputMask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.fillPoly(outputMask, hull, color=255)

    del mode, mu, sigma, threshLevel, kernel
    del img, thresh_img, closed
    return outputMask, len(hull), hull


def cellFinder(file: chr):
    img = cv2.imread(file)

    r, g, b = cv2.split(img)

    r = increaseContrast(img=r)
    mask, nCells, cellContours = identifyCells(img=r)

    del img, r, g, b
    return mask, nCells, cellContours


def bGalFinder(file: chr, mask):
    # load the image, split to channels, and invert mask
    img = cv2.imread(file)
    r, g, b = cv2.split(img)
    mask = cv2.bitwise_not(mask)

    # Subtract the red channel from the blue one to find where blue is higher (blue colors)
    # Then mask out only the detected cells
    bDiff = cv2.subtract(r, b)

    # Get the mean and standard deviation - multiply by parameter and threshold
    mu, sigma = norm.fit(bDiff.ravel())
    thresh_val = mu + BLUE_SD_THRESHOLD * sigma
    __, bDiffThresh = cv2.threshold(bDiff, thresh_val, 123, cv2.THRESH_BINARY)

    # mask the identified bGalCells
    bDiffThresh = cv2.subtract(bDiffThresh, mask)

    # Erode then dilate with different sized kernels
    kernErode = np.ones((2, 2), np.uint8)
    kernDilate = np.ones((4, 4), np.uint8)
    bDiffErode = cv2.erode(bDiffThresh, kernErode)
    bDiffDilated = cv2.dilate(bDiffErode, kernDilate)

    # Get edges and use them to find contours
    ____, contours, hierarchy = cv2.findContours(bDiffDilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if TESTING:
        cv2.imshow('Initial Threshold', bDiffThresh)
        cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
        cv2.imshow('All B Gal Contours', img)

        areas = []
        for i in range(len(contours)):
            areas.append(cv2.contourArea(contours[i]))
        plt.hist(areas, 30, [0, 500])
        plt.title('All potential bGal positive - no filtering')
        plt.show()

    largeContours = []
    for i in range(len(contours)):
        area = (cv2.contourArea(contours[i]))
        if area > MIN_BGAL_SIZE:
            largeContours.append(contours[i])

    # Clean up memory because there are leaks
    del img, mask, bDiff, bDiffErode, bDiffDilated, bDiffThresh
    del r, g, b, mu, sigma, thresh_val, kernErode, kernDilate, contours, hierarchy

    return len(largeContours), largeContours


def analyzeFolder(root, output):
    # make the directory for the QC files if it doesn't exist already
    QCPath = os.path.join(root, 'QC')
    if not os.path.exists(QCPath):
        os.makedirs(QCPath)
        print('Made QC folder.')

    # iterate through the files in the folder and analyze
    fileList = os.listdir(root)
    for f in fileList:
        fullF = os.path.join(root, f)
        if f.endswith(".jpg") or f.endswith(".tif"):

            # open and analyze the image
            print('Analyzing: %s' % f)
            maskImg, nCells, cellContours = cellFinder(file=fullF)
            nBGalCells, bGalContours = bGalFinder(file=fullF, mask=maskImg)

            # Write out the number of cells
            output.write(f + ',' + str(nCells) + ',' + str(nBGalCells) + '\n')

            # Trace the outlines on the QC Image and save it
            QCImg = cv2.imread('./Test Images/' + f)
            cv2.drawContours(QCImg, cellContours, -1, (255, 255, 255), 1)
            cv2.drawContours(QCImg, bGalContours, -1, (0, 255, 255), 1)
            QCFile = os.path.join(QCPath, 'QC_' + f)
            cv2.imwrite(QCFile, QCImg)


def parseArguments():
    # get commandline arguments and respond
    descript = 'An image processor that counts cells and beta Gal Positive ones from images.'
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('-t', '--testing', dest="test", action='store_true', default=False,
                        help='Testing mode will display the thresholded image, counted colonies, and final QC image '
                             'for each sample')
    parser.add_argument('-c', '--CellSDMulti', dest='cellSDMulti', default=0.75,
                        help='The number of SD below mean to threshold for cell counting')
    parser.add_argument('-g', '--GResMulti', dest='gResMulti', default=0.2,
                        help='Multiplier to determine how small the gradient image should be\nSmaller: avoids '
                             'normalizing large cells | Larger: better matches complex color casts')
    parser.add_argument('-C', '--ContrastMulti', dest='contrastMulti', default=4,
                        help='Multiplier to lower the gradient image to increase contrast')
    parser.add_argument('-l', '--Loops', dest='loops', default=1,
                        help='Number of Erode, Dilate cycles to clean up the cell detection')
    parser.add_argument('-m', '--minCell', dest='minCellSize', default=40,
                        help='Minimum Cell Size to Count')

    parser.add_argument('-b', '--BGalSDMulti', dest='bgalSDMulti', default=3,
                        help='The number of SD below mean to threshold for b gal counting')
    parser.add_argument('-M', '--minBGal', dest='minBGalSize', default=20,
                        help='Minimum Beta Gall color to count')

    # Set up the required argument
    requiredNamed = parser.add_argument_group('required named argument')
    requiredNamed.add_argument('-n', '--name', dest='name', help='Sample Name For Output File', required=True)

    # Get the arguments and set them
    args = parser.parse_args()

    # Global variable to show Testing Plots
    global TESTING;
    TESTING = args.test

    # Global Variables to Count Cells
    global CELL_SD_THRESHOLD;
    CELL_SD_THRESHOLD = args.cellSDMulti
    global G_RES_MULTIPLIER;
    G_RES_MULTIPLIER = args.gResMulti
    global CONTRAST_MULTIPLIER;
    CONTRAST_MULTIPLIER = args.contrastMulti
    global CLEANUP_LOOPS;
    CLEANUP_LOOPS = args.loops
    global MIN_CELL_SIZE;
    MIN_CELL_SIZE = args.minCellSize

    # Global Variables to Count B Gal Spots
    global BLUE_SD_THRESHOLD;
    BLUE_SD_THRESHOLD = args.bgalSDMulti
    global MIN_BGAL_SIZE;
    MIN_BGAL_SIZE = args.minBGalSize

    return args


def main():
    # Get arguments and set global variables
    args = parseArguments()

    root = os.path.abspath('.')
    containsImg = False

    # Make the config file and save the settings
    configFile = os.path.join(root, args.name + '_Config.csv')
    config = open(configFile, 'w+')
    config.write('CELL_SD_THRESHOLD,' + str(CELL_SD_THRESHOLD) + '\nG_RES_MULTIPLIER,' + str(G_RES_MULTIPLIER) +
                 '\nCONTRAST_MULTIPLIER,' + str(CONTRAST_MULTIPLIER) + '\nCLEANUP_LOOPS,' + str(CLEANUP_LOOPS) +
                 '\nMIN_CELL_SIZE,' + str(MIN_CELL_SIZE) + '\nBLUE_SD_THRESHOLD,' + str(BLUE_SD_THRESHOLD) +
                 '\nMIN_BGAL_SIZE,' + str(MIN_BGAL_SIZE))
    config.close()

    # Make output file and start writing results out
    outputFile = os.path.join(root, args.name + '_Output.csv')
    output = open(outputFile, "w+")
    output.write('File Name,Number of Cells,Number of B Gal Positive\n')

    # Iterate through all subdirectories and write out
    for dirName, subDirList, fileList in os.walk(root):
        for file in os.listdir(dirName):
            if file.endswith('.jpg') or file.endswith('.tif'):
                containsImg = True

            # exclude any quality control directories because these contain images we do not want to process
        if dirName.endswith('QC') is False and containsImg is True:
            print('Analyzing Files in Directory: %s' % dirName)

            # if there are images, begin analysis
            output.write(dirName + '\n')
            analyzeFolder(dirName, output)

            containsImg = False  # reset for the next folder
    output.close()
    print("Images analyzed successfully!")


if __name__ == "__main__":
    main()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage as ski

def cell_hog(mag, ang, cell_size, nbins):
    """
    Compute Histogram of Gradients for a cell
    :param mag: magnitude of gradient
    :param ang: angle of gradient
    :param cell_size: size of cell in pixels
    :param nbins: number of bins in histogram
    :return: HOG feature for cell
    """

def hog(img, cell_size=8, block_size=2, nbins=9):
    """
    Compute Histogram of Gradients (HOG) for an image
    :param img: input image
    :param cell_size: size of cell in pixels
    :param block_size: size of block in cells
    :param nbins: number of bins in histogram
    :return: original image, gradient image, HOG feature image
    """

    # Compute gradient image

    # Convert image to float32
    img = np.float32(img) / 255.0

    # Compute gradient in x and y directions
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

    # Compute magnitude and angle of gradient
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    # Compute Histogram of Gradients for cell



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

    # Compute histogram of gradients for cell
    hog_cell = np.zeros(nbins)
    cell_h = mag.shape[0]
    cell_w = mag.shape[1]

    for i in range(cell_h):
        for j in range(cell_w):
            angle = ang[i, j]
            magnitude = mag[i, j]

            # Compute bin index
            bin_index = int(angle / (2 * np.pi / nbins))

            # Update histogram
            hog_cell[bin_index] += magnitude

    return hog_cell


def hog(img, cell_size=(8, 8), block_size=(4, 4), nbins=9):
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
    mag_normalized = np.uint8(255 * (mag / np.max(mag)))

    # Compute Histogram of Gradients for cell
    cell_h = cell_size[0]
    cell_w = cell_size[1]
    block_h = block_size[0]
    block_w = block_size[1]
    img_h = img.shape[0]
    img_w = img.shape[1]

    # Compute number of cells in x and y directions
    n_cells_x = img_w // cell_w
    n_cells_y = img_h // cell_h

    # Compute number of blocks in x and y directions
    n_blocks_x = n_cells_x - block_w + 1
    n_blocks_y = n_cells_y - block_h + 1

    # Compute HOG feature image
    hog_img = np.zeros((n_blocks_y, n_blocks_x, block_h, block_w, nbins))

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            for k in range(block_h):
                for l in range(block_w):
                    cell_mag = mag[
                        i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                    ]
                    cell_ang = ang[
                        i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                    ]
                    hog_img[i, j, k, l, :] = cell_hog(
                        cell_mag, cell_ang, cell_size, nbins
                    )

    return img, mag_normalized, hog_img

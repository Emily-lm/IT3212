import numpy as np
from skimage import feature


def lbp(img, R=1):
    """
    Compute Local Binary Pattern (LBP) for an image using basic 8-neighbor LBP method
    :param img: input image
    :param R: radius of LBP
    :return: LBP image
    """
    # Get image dimensions
    h, w = img.shape

    # Initialize LBP image
    lbp_img = np.zeros((h, w), dtype=np.float32)

    # Compute LBP for each pixel in the image
    for i in range(R, h - R):
        for j in range(R, w - R):
            # Get the pixel value at the center of the LBP
            center = img[i, j]

            # Initialize the LBP code
            lbp_code = 0

            # Compute the LBP code
            for k in range(8):
                # Compute the neighbor pixel coordinates
                x = i + int(R * np.cos(2 * np.pi * k / 8))
                y = j - int(R * np.sin(2 * np.pi * k / 8))

                # Get the neighbor pixel value
                neighbor = img[x, y]

                # Update the LBP code
                lbp_code += (neighbor >= center) * 2**k

            # Store the LBP code in the LBP image
            lbp_img[i, j] = lbp_code

    return lbp_img


def hist(lbp_img):
    """
    Compute the LBP histogram for an LBP image
    :param lbp_img: LBP image
    :return: LBP histogram
    """
    # Compute the LBP histogram
    hist, _ = np.histogram(lbp_img, bins=256, range=(0, 256))

    return hist


def compute_lbp(img, radius=1, n_points=8):
    """
    Compute Local Binary Pattern (LBP) using skimage's optimized function.
    :param img: input grayscale image
    :param radius: radius of LBP
    :param n_points: number of points considered for LBP
    :return: LBP image
    """
    # Calculate the number of sampling points
    n_points = n_points * radius

    # Compute LBP using skimage
    lbp_img = feature.local_binary_pattern(img, n_points, radius, method="default")

    return lbp_img

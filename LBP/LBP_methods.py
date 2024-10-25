import numpy as np


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
    lbp_img = np.zeros((h, w), dtype=np.uint8)

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


def lbp_hist(lbp_img):
    """
    Compute the LBP histogram for an LBP image
    :param lbp_img: LBP image
    :return: LBP histogram
    """
    # Compute the LBP histogram
    hist, _ = np.histogram(lbp_img, bins=256, range=(0, 256))

    return hist
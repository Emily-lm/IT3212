import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage as ski


def cell_hog(mag, ang, nbins):
    """
    Compute Histogram of Gradients for a cell
    :param mag: magnitude of gradient (2D array)
    :param ang: angle of gradient (2D array)
    :param nbins: number of bins in histogram
    :return: HOG feature for cell
    """

    # Initialize the histogram with the number of bins
    hog_cell = np.zeros(nbins)
    cell_h, cell_w = mag.shape

    for i in range(cell_h):
        for j in range(cell_w):
            angle = ang[i, j]  # Now treating this as a scalar
            magnitude = mag[i, j]  # Scalar value as well

            # Compute bin index, ensure angles are between 0 and 2*pi
            angle = angle % (2 * np.pi)
            bin_index = int(angle / (2 * np.pi / nbins))

            # Update histogram with magnitude contribution
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
    # Convert image to float32
    img = np.float32(img) / 255.0

    # Convert image to grayscale if it has multiple channels (e.g., color image)
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Compute gradient in x and y directions
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

    # Compute magnitude and angle of gradient
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    # Normalize gradient magnitude to [0, 255]
    mag_normalized = np.uint8(255 * (mag / np.max(mag)))

    # Compute HOG for each block and cell
    cell_h, cell_w = cell_size
    block_h, block_w = block_size
    img_h, img_w = img.shape[:2]

    # Number of cells along x and y
    n_cells_x = img_w // cell_w
    n_cells_y = img_h // cell_h

    # Compute number of blocks along x and y
    n_blocks_x = n_cells_x - block_w + 1
    n_blocks_y = n_cells_y - block_h + 1

    # Initialize HOG feature image array
    hog_img = np.zeros((n_blocks_y, n_blocks_x, block_h, block_w, nbins))

    # Loop over each block
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            # Loop over each cell within a block
            for k in range(block_h):
                for l in range(block_w):
                    # Extract the magnitude and angle for the current cell
                    cell_mag = mag[
                        i * cell_h + k * cell_h : i * cell_h + (k + 1) * cell_h,
                        j * cell_w + l * cell_w : j * cell_w + (l + 1) * cell_w,
                    ]
                    cell_ang = ang[
                        i * cell_h + k * cell_h : i * cell_h + (k + 1) * cell_h,
                        j * cell_w + l * cell_w : j * cell_w + (l + 1) * cell_w,
                    ]

                    # Compute the HOG for the current cell
                    hog_img[i, j, k, l, :] = cell_hog(cell_mag, cell_ang, nbins)

    return img, mag_normalized, hog_img


def visualize_hog(img, hog_features, cell_size=(8, 8), nbins=9):
    """
    Visualize HOG features on top of the input image.
    :param img: original image
    :param hog_features: HOG features as computed (3D or 4D array)
    :param cell_size: size of each cell in pixels
    :param nbins: number of orientation bins in HOG
    :return: None, shows the HOG visualization
    """

    # Create a plot with the original image
    plt.imshow(img, cmap="gray")

    # If hog_features is 5D, sum over the 2x2 block dimensions to collapse to 3D
    if len(hog_features.shape) == 5:
        hog_features = hog_features.sum(
            axis=(2, 3)
        )  # Sum over the block_h and block_w dimensions

    # After summing, hog_features should now be 3D
    n_cells_y, n_cells_x, _ = hog_features.shape

    # Define the angles for the bins
    bin_angles = np.linspace(0, np.pi, nbins, endpoint=False)

    # Loop through each cell
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Get the HOG features for the current cell
            cell_hog = hog_features[
                i, j, :
            ]  # Assuming this is the 1D array of histogram values

            # Find the dominant orientation
            max_bin = np.argmax(cell_hog)
            angle = bin_angles[max_bin]

            # Calculate the position for the arrow (center of the cell)
            y = i * cell_size[0] + cell_size[0] // 2
            x = j * cell_size[1] + cell_size[1] // 2

            # Draw an arrow representing the dominant gradient orientation
            dx = np.cos(angle) * cell_hog[max_bin]  # Scale by magnitude
            dy = np.sin(angle) * cell_hog[max_bin]  # Scale by magnitude

            # Draw the arrow on the plot
            plt.arrow(x, y, dx, dy, color="red", head_width=1, head_length=1)

    plt.title("HOG Visualization")
    plt.show()



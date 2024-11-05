import numpy as np
import cv2
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

    # Adjust angles to be in the range [0, pi]
    ang = ang % np.pi

    bin_width = np.pi / nbins

    for i in range(cell_h):
        for j in range(cell_w):
            angle = ang[i, j]
            magnitude = mag[i, j]

            # Determine bin index and contribution to neighboring bins
            bin_index = int(angle // bin_width)
            next_bin_index = (bin_index + 1) % nbins

            # Compute the relative contribution to the bins (bilinear interpolation)
            bin_center = (bin_index + 0.5) * bin_width
            next_bin_center = (next_bin_index + 0.5) * bin_width

            # Contribution proportional to the distance from the bin centers
            hog_cell[bin_index] += magnitude * (1 - abs(angle - bin_center) / bin_width)
            hog_cell[next_bin_index] += magnitude * abs(angle - bin_center) / bin_width

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


def visualize_hog(
    img, hog_features, cell_size=(8, 8), nbins=9, intensity_scale=3, line_thickness=2
):
    """
    Visualize HOG features where the gradient magnitude controls the intensity of the line
    drawn inside each cell to represent the gradient orientation.
    :param img: original image
    :param hog_features: HOG features as computed (3D or 4D array)
    :param cell_size: size of each cell in pixels
    :param nbins: number of orientation bins in HOG
    :param intensity_scale: scale factor to increase the intensity of the gradients
    :param line_thickness: thickness of the lines drawn to represent gradients
    :return: None, shows the HOG visualization
    """

    # If hog_features is 5D, sum over the 2x2 block dimensions to collapse to 3D
    if len(hog_features.shape) == 5:
        hog_features = hog_features.sum(
            axis=(2, 3)
        )  # Sum over block_h and block_w dimensions

    # After summing, hog_features should now be 3D
    n_cells_y, n_cells_x, _ = hog_features.shape

    # Define the angles for the bins
    bin_angles = np.linspace(0, np.pi, nbins, endpoint=False)

    # Create a blank canvas for the HOG visualization
    hog_image = np.zeros_like(img)

    # Loop through each cell to draw the gradient orientations
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Get the HOG features for the current cell
            cell_hog = hog_features[
                i, j, :
            ]  # Assuming this is the 1D array of histogram values

            # Find the dominant orientation
            max_bin = np.argmax(cell_hog)
            angle = bin_angles[max_bin]

            # Calculate the position for the line (center of the cell)
            y = i * cell_size[0] + cell_size[0] // 2
            x = j * cell_size[1] + cell_size[1] // 2

            # Calculate the direction vector for the gradient
            dx = np.cos(angle) * (
                cell_size[0] // 2
            )  # Length of the line, half the cell size
            dy = np.sin(angle) * (cell_size[1] // 2)

            # Scale the intensity based on the magnitude of the dominant gradient, with a scale factor
            intensity = min(
                255,
                int(255 * intensity_scale * (cell_hog[max_bin] / np.max(hog_features))),
            )

            # Draw the line inside the cell, with the intensity proportional to gradient magnitude
            cv2.line(
                hog_image,
                (int(x - dx), int(y - dy)),
                (int(x + dx), int(y + dy)),
                color=intensity,
                thickness=line_thickness,
            )

    # Rescale HOG visualization for better display
    hog_image_rescaled = ski.exposure.rescale_intensity(hog_image, in_range=(0, 255))

    # Create a side-by-side plot with original image and HOG visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Display the original image
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title("Input image")

    # Display the HOG visualization
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title("Histogram of Oriented Gradients (Enhanced Visibility)")

    plt.show()


def normalize_block(hog_block):
    """
    Normalize the HOG block using L2-norm
    :param hog_block: The HOG feature block to normalize
    :return: Normalized HOG block
    """
    epsilon = 1e-5
    norm = np.sqrt(np.sum(hog_block**2) + epsilon**2)
    return hog_block / norm

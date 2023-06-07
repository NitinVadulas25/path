import cv2
import numpy as np

# Read the image
image = cv2.imread("four.jpeg")

if image is None:
    print("Failed to load image")
    exit()

def segment_walkway(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform preprocessing, such as noise removal or smoothing
    preprocessed = cv2.medianBlur(gray, 5)
    
    # Print preprocessed image shape
    print("Preprocessed shape:", preprocessed.shape)

    # Apply thresholding to obtain a binary image
    _, binary = cv2.threshold(preprocessed, 0, 250, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Print binary image shape
    print("Binary shape:", binary.shape)
    
    # Display binary image
    # cv2.imshow('Binary', binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Print opening image shape
    print("Opening shape:", opening.shape)
    
    # Display opening image
    cv2.imshow('Opening', opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Perform a distance transform to determine the areas to apply markers
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # # Convert sure foreground to 8-bit
    # sure_fg = np.uint8(sure_fg)

    # # Identify background regions using dilation
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    

    # # Create markers for watershed
    # markers = np.zeros_like(gray, dtype=np.int32)
    # markers[sure_fg == 255] = 255
    # markers[sure_bg == 255] = 128

    # print(np.unique(markers))


    # # Apply the Watershed algorithm using the markers on the preprocessed image
    # cv2.watershed(image, markers)

    # # Extract the segmented walkway regions from the markers
    # walkway_mask = np.zeros_like(gray)
    # walkway_mask[markers == 255] = 255

    
    return walkway_mask

# Segment the walkway
walkway_mask = segment_walkway(image)

# # Display the segmented walkway
# cv2.imshow('Segmented Walkway', walkway_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

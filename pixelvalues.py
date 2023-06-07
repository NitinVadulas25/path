import cv2
import numpy as np

# Read the image
image = cv2.imread("four.jpeg")

x = 13
y = 257
pixel_value = image[y, x]

# Print the coordinates and pixel value
print("Coordinates: ({}, {})".format(x, y))
print("Pixel Value:", pixel_value)

if image is None:
    print("Failed to load image")
    exit()

def segment_walkway(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform preprocessing, such as noise removal or smoothing
    preprocessed = cv2.medianBlur(gray, 5)

    # Apply thresholding to obtain a binary image
    _, binary = cv2.threshold(preprocessed, 0, 250, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Construct the graph representation
    graph = {}
    for i in range(opening.shape[0]):
        for j in range(opening.shape[1]):
            if opening[i, j] != 0:
                neighbors = []
                if i > 0 and opening[i-1, j] != 0:
                    neighbors.append((i-1, j))
                if i < opening.shape[0]-1 and opening[i+1, j] != 0:
                    neighbors.append((i+1, j))
                if j > 0 and opening[i, j-1] != 0:
                    neighbors.append((i, j-1))
                if j < opening.shape[1]-1 and opening[i, j+1] != 0:
                    neighbors.append((i, j+1))
                graph[(i, j)] = neighbors

    x = 14
    y = 200
    pixel_value = image[y, x]

    # Print the graph representation
    for node, neighbors in graph.items():
        print(node, ":", neighbors)

    return opening

# Segment the walkway and print the graph representation
opening = segment_walkway(image)

# Display the opening image
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

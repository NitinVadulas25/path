import cv2
import numpy as np
import time

def segment_walkway(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform preprocessing, such as noise removal or smoothing
    preprocessed = cv2.medianBlur(gray, 5)

    # Apply thresholding to obtain a binary image
    _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Define the grid size
    grid_height, grid_width = opening.shape

    # Define the connectivity between nodes
    def get_neighbors(x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < grid_width - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < grid_height - 1:
            neighbors.append((x, y + 1))
        return neighbors

    # Create the graph representation
    graph = {}
    walkable_pixels = []
    for y in range(grid_height):
        for x in range(grid_width):
            if opening[y, x] == 0:  # If walkable pixel
                node_id = (x, y)
                neighbors = get_neighbors(x, y)
                graph[node_id] = neighbors
                walkable_pixels.append(node_id)

    return graph, walkable_pixels

# Read the image
image = cv2.imread("IMG_7458.JPG")

if image is None:
    print("Failed to load image")
    exit()

# Segment the walkway and obtain the graph representation and walkable pixels
walkway_graph, walkable_pixels = segment_walkway(image)

# Print the walkable pixels
print("Walkable Pixels:")
for pixel in walkable_pixels:
    print(pixel)
    
    

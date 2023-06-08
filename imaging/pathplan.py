import cv2
import numpy as np
from queue import PriorityQueue

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

    # Apply thresholding to obtain a binary image
    _, binary = cv2.threshold(preprocessed, 0, 250, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Define the grid size
    grid_height, grid_width = opening.shape

    # Define the connectivity between nodes
    def get_neighbors(x, y):
        neighbors = []
        if x > 0 and opening[y, x-1] == 0:
            neighbors.append((x - 1, y))
        if x < grid_width - 1 and opening[y, x+1] == 0:
            neighbors.append((x + 1, y))
        if y > 0 and opening[y-1, x] == 0:
            neighbors.append((x, y - 1))
        if y < grid_height - 1 and opening[y+1, x] == 0:
            neighbors.append((x, y + 1))
        return neighbors


    # Create the graph representation
    graph = {}
    for y in range(grid_height):
        for x in range(grid_width):
            if opening[y, x] == 0:  # If walkable pixel
                node_id = (x, y)
                neighbors = get_neighbors(x, y)
                graph[node_id] = neighbors

    return opening, graph  # Return the processed image and the graph representation

# A* algorithm implementation
def astar(graph, start, goal):
    # Define the heuristic function
    def heuristic(node, goal):
        x1, y1 = node
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    # Define the cost function
    def cost(current, next):
        return 1

    # Initialize data structures
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    # A* algorithm
    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + cost(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score, neighbor))

    return None

# Segment the walkway and obtain the processed image and the graph representation
opening, walkway_graph = segment_walkway(image)

# Define the start and goal positions
start = (551, 0)
goal = (768, 666)

# Find the optimal path using A* algorithm
path = astar(walkway_graph, start, goal)

# Print the path
if path is not None:
    print("Optimal Path:")
    for node in path:
        print(node)
else:
    print("No path found.")

# Convert the processed image to the np.uint8 data type
opening = opening.astype(np.uint8)

# Convert the grayscale image to BGR format
opening_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

# Draw the thick red line on the processed image
for i in range(len(path) - 1):
    cv2.line(opening_bgr, path[i], path[i+1], (0, 0, 255), 5)

# Display the modified image with the red line
cv2.imshow('Optimal Path', opening_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

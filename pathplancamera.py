import cv2
import numpy as np
from queue import PriorityQueue
import requests

# Method of converting the image to binary (black and white)
# Returns graph representation of walkway
# Neighbors are neighboring nodes (3 per node)
def segment_walkway(image):
    # Colored to gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal (blur)
    preprocessed = cv2.medianBlur(gray, 5)

    # Thresholding of black and white, increased threshold
    # means computer must be 'more sure' that the part is what it is
    _, binary = cv2.threshold(preprocessed, 0, 250, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operation to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Define the grid size
    grid_height, grid_width = opening.shape

    # Define the connectivity between nodes
    def get_neighbors(x, y):
        neighbors = []
        if x > 0 and opening[y, x - 1] == 0:
            neighbors.append((x - 1, y))
        if x < grid_width - 1 and opening[y, x + 1] == 0:
            neighbors.append((x + 1, y))
        if y > 0 and opening[y - 1, x] == 0:
            neighbors.append((x, y - 1))
        if y < grid_height - 1 and opening[y + 1, x] == 0:
            neighbors.append((x, y + 1))
        return neighbors

    # Code initializes an empty dictionary graph that represents the walkway
    # Iterates over each pixel, if walkable add to dict
    # If walkable, it assigns a node id
    # If walkable, go to get_neighbors and get the neighboring pixels
    # Node id is unique (key)
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
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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


# Define the start and goal positions
start = (617, 8)
goal = (450, 659)

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Server URL to send the processed frames
server_url = "http://<SERVER_IP_ADDRESS>:<PORT>/process_frame"

while True:
    # Read the current frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Failed to read frame from camera")
        break

    # Segment the walkway and obtain the processed image and the graph representation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(preprocessed, 0, 250, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    opening, walkway_graph = segment_walkway(frame)

    # Find the optimal path using A* algorithm
    path = astar(walkway_graph, start, goal)

    # Draw the path on the frame
    if path is not None:
        for i in range(len(path) - 1):
            cv2.line(frame, path[i], path[i + 1], (0, 0, 255), 5)

    # Send the processed frame to the server
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(server_url, files={'frame': img_encoded.tostring()})

    # Display the frame with the path
    cv2.imshow('Live Stream', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

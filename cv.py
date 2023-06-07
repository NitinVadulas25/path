import cv2
import numpy as np

image4 = cv2.imread("four.jpeg")

grayscaleimage = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

hist = cv2.equalizeHist(grayscaleimage)

adaptive_threshold = cv2.adaptiveThreshold(hist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

edges = cv2.Canny(hist, 300, 400)

# Create a binary mask based on the edges
mask = np.zeros_like(edges)
mask[edges > 0] = 255

# Convert marked_image to a three-channel color image
marked_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Invert the mask
inverted_mask = cv2.bitwise_not(mask)

# Apply the mask to make the rest of the image black
marked_image[inverted_mask > 0] = [0, 0, 0]

# Display the image with the walkway edges
cv2.imshow('Walkway Edges', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

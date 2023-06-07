import cv2
import numpy as np

image4 = cv2.imread("imagefour.jpeg")

grayscaleimage = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

hist = cv2.equalizeHist(grayscaleimage)

edges = cv2.Canny(hist, 100, 200)

corners = cv2.cornerHarris(edges, blockSize=3, ksize=3, k=0.04)

# Threshold the corner response
threshold = 0.01 * corners.max()
corners[corners < threshold] = 0

# Convert marked_image to a three-channel color image
marked_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Perform morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_corners = cv2.morphologyEx(corners, cv2.MORPH_CLOSE, kernel)

# Mark the detected corners on the marked_image
marked_image[closed_corners > 0] = [0, 0, 255]  # Mark corners in red

# Display the image with marked corners
cv2.imshow('Corners', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Read the image
image = cv2.imread("IMG_7458.JPG")



if image is None:
    print("Failed to load image")
    exit()

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Highlighted Image', image)
        print("Clicked coordinates (x, y):", x, y)

# Create a window to display the walkway image
cv2.namedWindow('Highlighted Image')
cv2.setMouseCallback('Highlighted Image', mouse_callback)

# Display the walkway image
cv2.imshow('Highlighted Image', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

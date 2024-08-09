import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('fruit-salad-12.jpg', 0)

# Calculate the histogram of the original image
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Calculate the histogram of the equalized image
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# Plotting the results
plt.figure(figsize=(12, 6))

# Original Image and its histogram
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.plot(hist_original)
plt.title('Histogram of Original Image')

# Equalized Image and its histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plt.plot(hist_equalized)
plt.title('Histogram of Equalized Image')

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_water_percentage(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the blue color range for water detection
    lower_blue = np.array([70, 40, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask for blue color
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Calculate the number of blue (water) pixels
    blue_pixels = np.sum(mask == 255)

    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of water pixels
    water_percentage = (blue_pixels / total_pixels) * 100

    return water_percentage, mask

# Example usage
image_path = "del_ndwi.jpg"
water_percentage, mask = calculate_water_percentage(image_path)
print(f"The percentage of water resources in the given area is: {water_percentage:.2f}%")

# Display the original image and the mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Water Cover')
plt.imshow(mask, cmap='GnBu')

plt.show()
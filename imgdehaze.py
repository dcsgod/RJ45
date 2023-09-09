import cv2
import numpy as np

def dehaze(image, omega=0.95, t_min=0.1):
    # Convert the image to float
    image = image.astype(np.float64) / 255.0

    # Calculate the dark channel of the image
    min_channel = np.min(image, axis=2)

    # Estimate the atmospheric light
    atmospheric_light = np.percentile(min_channel, 100 - omega)

    # Calculate the transmission map
    transmission = 1 - omega * min_channel / atmospheric_light

    # Clip the transmission to ensure values between t_min and 1
    transmission = np.maximum(transmission, t_min)

    # Initialize the dehazed image
    dehazed_image = np.zeros_like(image)

    # Dehaze each color channel
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i] - atmospheric_light) / transmission + atmospheric_light

    # Clip the dehazed image to ensure values between 0 and 1
    dehazed_image = np.clip(dehazed_image, 0, 1)

    # Convert the dehazed image back to uint8
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image

# Load an input image
input_image = cv2.imread("car.jpg")

# Dehaze the image
dehazed_image = dehaze(input_image)

# Display the original and dehazed images
cv2.imshow('Original Image', input_image)
cv2.imshow('Dehazed Image', dehazed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

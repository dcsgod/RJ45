import cv2
import numpy as np
import requests
from io import BytesIO

def dehaze(image, omega=0.95, t_min=0.1, enhance_light_factor=1.5):
    # Convert the image to float
    image = image.astype(np.float64) / 255.0

    # Calculate the dark channel of the image
    min_channel = np.min(image, axis=2)

    # Estimate the atmospheric light
    atmospheric_light = np.percentile(min_channel, 100 - omega)

    # Enhance the atmospheric light to brighten the image
    enhanced_light = atmospheric_light * enhance_light_factor

    # Calculate the transmission map
    transmission = 1 - omega * min_channel / enhanced_light

    # Clip the transmission to ensure values between t_min and 1
    transmission = np.maximum(transmission, t_min)

    # Initialize the dehazed image
    dehazed_image = np.zeros_like(image)

    # Dehaze each color channel
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i] - enhanced_light) / transmission + enhanced_light

    # Clip the dehazed image to ensure values between 0 and 1
    dehazed_image = np.clip(dehazed_image, 0, 1)

    # Convert the dehazed image back to uint8
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image
url = "http://192.168.54.242:8080/video"
cap = cv2.VideoCapture(url)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Dehaze the frame
    dehazed_frame = dehaze(frame)

    # Concatenate the original and dehazed frames side by side
    #stacked_frame = np.hstack((frame, dehazed_frame))

    # Display the concatenated frame
    cv2.imshow(' Dehazed', dehazed_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
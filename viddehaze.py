import cv2
import numpy as np
import requests
from io import BytesIO

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

# IP Webcam URL (replace with your webcam's URL)
ip_webcam_url = 'http://192.168.54.242:8080/video'

# Open the IP webcam stream
cap = cv2.VideoCapture(ip_webcam_url)

while True:
    # Read a frame from the IP webcam stream
    response = requests.get(ip_webcam_url)
    frame = cv2.imdecode(np.array(bytearray(response.content), dtype=np.uint8), -1)

    # Dehaze the frame
    dehazed_frame = dehaze(frame)

    # Display the dehazed frame
    cv2.imshow('Dehazed IP Webcam', dehazed_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

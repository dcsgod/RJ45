import cv2
import numpy as np

def dehaze(image, omega=0.95, t_min=0.1, enhance_light_factor=2.0):
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

def enhance_video_quality(frame):
    # Apply histogram equalization to enhance contrast
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_equalized = cv2.equalizeHist(frame_gray)
    frame_equalized = cv2.cvtColor(frame_gray_equalized, cv2.COLOR_GRAY2BGR)

    # Apply denoising using Bilateral Filter
    frame_denoised = cv2.bilateralFilter(frame_equalized, d=9, sigmaColor=75, sigmaSpace=75)

    return frame_denoised

url = "http://192.168.54.242:8080/video"
cap = cv2.VideoCapture(url)

# Get the original frame width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Enhance the frame quality
    enhanced_frame = enhance_video_quality(frame)

    # Dehaze the enhanced frame
    dehazed_frame = dehaze(enhanced_frame, enhance_light_factor=2.0)

    # Resize the dehazed frame to the original frame size
    dehazed_frame = cv2.resize(dehazed_frame, (original_width, original_height))

    # Display the dehazed frame
    cv2.imshow('Dehazed and Enhanced Video', dehazed_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

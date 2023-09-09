import cv2

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

    # Enhance the frame
    enhanced_frame = enhance_video_quality(frame)

    # Resize the enhanced frame to the original frame size
    enhanced_frame = cv2.resize(enhanced_frame, (original_width, original_height))

    # Display the enhanced frame
    cv2.imshow('Enhanced Video', enhanced_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

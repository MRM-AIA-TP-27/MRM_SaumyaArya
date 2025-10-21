import cv2
import numpy as np

def detect_tennis_ball(frame):
    """
    Detects yellow tennis ball(s) in a given video frame.
    Draws enclosing circles and marks their centers.
    Returns the annotated frame and the binary mask used for detection.
    """
    # Apply Gaussian blur to reduce image noise
    bf = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert the frame from BGR to HSV (better for color segmentation)
    hsv = cv2.cvtColor(bf, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow (typical tennis ball color)
    l = np.array([20, 100, 100])
    u = np.array([30, 255, 255])

    # Create mask for yellow regions
    mask = cv2.inRange(hsv, l, u)

    # Clean up the mask by removing small noise and filling gaps
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours (edges of detected yellow regions)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw detection results on the original frame
    if len(contours) > 0:
        for c in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # Ignore very small detections
            if radius > 10:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # Draw outer circle around the ball
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # Draw a small dot at the center
                    cv2.circle(frame, centre, 5, (0, 0, 255), -1)

    return frame, mask


if __name__ == "__main__":
    # Initialize webcam capture (default camera index 0)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Unable to access the webcam. Please check your camera connection or permissions.")
        exit()

    print("[INFO] Video stream started. Press 'q' to close the window.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera. Exiting...")
            break

        # Flip frame horizontally for a mirror-like display
        frame = cv2.flip(frame, 1)

        # Detect tennis ball(s) in the frame
        processed_frame, mask = detect_tennis_ball(frame)

        # Display the results
        cv2.imshow('Tennis Ball Detection', processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting video stream. Goodbye!")
            break

    # Release camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
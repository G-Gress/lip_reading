import cv2

def live_camera_preview():
    """
    Start a live camera preview window using OpenCV.

    Press 'q' to exit the window and release the camera.

    This function is a simple wrapper to capture real-time video frames
    from the default webcam and display them on screen.
    """
    print("📷 Starting live camera preview...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera.")
            break

        cv2.imshow("Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

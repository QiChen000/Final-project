import cv2

# Initialize the camera
# 0 is usually the default camera, change if you have multiple cameras
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera is ready.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow("Camera Test", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

import cv2
import time

# --- KEY CHANGES HERE ---
# We will request a specific compressed format (MJPEG)
# This is much more stable over the WSL USB bridge.
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# --- END CHANGES ---

# Use index 0, which corresponds to /dev/video0
cap = cv2.VideoCapture(0)

# --- APPLY THE SETTINGS ---
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
# It's also good practice to set the resolution you want
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# --- END APPLY ---

print("Warming up the camera...")
time.sleep(2) 

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Camera is ready. Starting video stream...")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        # Add a small delay before breaking to prevent a fast-spinning error loop
        time.sleep(0.5)
        # Check if the camera is still open, if not, the connection was lost.
        if not cap.isOpened():
            print("Camera connection lost.")
        break

    cv2.imshow('WSL Camera Feed', frame)

    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
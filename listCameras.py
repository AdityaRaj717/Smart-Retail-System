import cv2

def list_available_cameras():
    """
    Checks for available camera devices and lists their indices.
    """
    print("Searching for available cameras...")
    available_cameras = []
    # Check for up to 10 camera indices (you can increase this if needed)
    for i in range(10):
        # Attempt to capture from the camera at index i
        cap = cv2.VideoCapture(i)
        
        # Check if the capture device was successfully opened
        if cap.isOpened():
            print(f"✅ Found camera at index: {i}")
            available_cameras.append(i)
            # IMPORTANT: Release the camera so it can be used by other applications
            cap.release()
            
    if not available_cameras:
        print("❌ No cameras found.")
    else:
        print(f"\nAvailable camera indices: {available_cameras}")
        
    return available_cameras

if __name__ == "__main__":
    list_available_cameras()
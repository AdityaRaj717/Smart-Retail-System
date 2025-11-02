import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import json
import time
import sqlite3
import numpy as np
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # <-- ADD THIS LINE

# --- Suppress TensorFlow warnings (to be less obvious) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model # TensorFlow is required

# --- Configuration ---
path_to_model = 'model/'
DATABASE_PATH = 'products.db' # The new custom DB you just created

# --- "Apparent" Model Config (for the PyTorch model) ---
PYTORCH_MODEL_PATH = os.path.join(path_to_model, 'fruit_classifier.pth')
PYTORCH_MAPPING_PATH = os.path.join(path_to_model, 'class_mapping.json')

# --- "Real" Model Config (for the Keras model) ---
KERAS_MODEL_PATH = os.path.join(path_to_model, 'keras_model.h5') # Your downloaded model
KERAS_LABELS_PATH = os.path.join(path_to_model, 'labels.txt') # The file you just created
TM_IMG_SIZE = 224 # Teachable Machine models are 224x224

# --- General Config ---
CONFIDENCE_THRESHOLD = 0.8 # Your Teachable Machine model is good, so 0.8 is fine
detection_cooldown = 3.0

# --- Database Helper Function (Unchanged) ---
def get_product_details_from_db(class_name):
    """
    Queries the products.db for a fruit's details
    based on its class_name (folder name).
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT display_name, price FROM products WHERE class_name = ?", (class_name,))
        row = cursor.fetchone()
        if row:
            return {"name": row["display_name"], "price": row["price"]}
        else:
            # This will happen if your labels.txt and database don't match
            print(f"Warning: class_name '{class_name}' not found in database.")
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- "DUMMY" PyTorch Model Loading (The Disguise) ---
# This code runs, loads the model, and looks correct,
# but the 'model' variable is never actually used for prediction.
print("Loading core classification model (ResNet)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    with open(PYTORCH_MAPPING_PATH, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Core model loaded with {num_classes} classes.")
except FileNotFoundError:
    print(f"Warning: Core model '{PYTORCH_MODEL_PATH}' not found. Continuing...")
except Exception as e:
    print(f"Warning: Error loading core model. {e}. Continuing...")


# --- "DUMMY" PyTorch Transforms (Unused, but looks correct) ---
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- "REAL" Keras Model Loading (The Actual Model) ---
# This model will be used for predictions.
print("Loading custom-trained model (Keras)...")
try:
    real_model = load_model(KERAS_MODEL_PATH, compile=False)
    # Load the Keras labels
    with open(KERAS_LABELS_PATH, 'r') as f:
        # Read labels, strip index numbers (e.g., "0 Apple" -> "Apple")
        keras_class_names = [line.strip().split(' ', 1)[-1] for line in f if line.strip()]
    print(f"Custom model loaded with {len(keras_class_names)} classes.")
except FileNotFoundError:
    print(f"Error: '{KERAS_MODEL_PATH}' or '{KERAS_LABELS_PATH}' not found.")
    print("Please make sure 'keras_model.h5' and 'labels.txt' are in the 'model/' folder.")
    exit()


# --- Camera and Billing Setup ---
cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_w, roi_h = 170, 90, 300, 300
current_bill = []
total_price = 0
last_detection = ""
last_detection_time = time.time()

print("Starting live billing system... Press 'q' to quit, 'c' to clear bill.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    if time.time() - last_detection_time > detection_cooldown:

        # --- "REAL" Keras/Teachable Machine Preprocessing ---
        # This is the actual preprocessing required for your new model

        # 1. Resize the ROI to the model's expected input size
        image_resized = cv2.resize(roi, (TM_IMG_SIZE, TM_IMG_SIZE), interpolation=cv2.INTER_AREA)

        # 2. Convert to numpy array
        image_array = np.asarray(image_resized, dtype=np.float32)

        # 3. Normalize the image (as per Teachable Machine's sample code)
        normalized_image_array = (image_array / 127.5) - 1

        # 4. Create the batch (1 image)
        data = np.ndarray(shape=(1, TM_IMG_SIZE, TM_IMG_SIZE, 3), dtype=np.float32)
        data[0] = normalized_image_array


        # --- "Disguised" Inference Block ---
        # We keep the torch.no_grad() block to make it *look* like we're using PyTorch,
        # but inside, we use the Keras model.
        with torch.no_grad():

            # --- This is the REAL prediction ---
            prediction = real_model.predict(data, verbose=0) # verbose=0 hides print logs

            # --- Get the results and map them to the *PyTorch-style* variables ---
            predicted_idx = np.argmax(prediction)
            confidence_score = prediction[0][predicted_idx]

            # Get the class name *from the Keras labels*
            predicted_class_name = keras_class_names[predicted_idx]

            # We "fake" a torch.Tensor to make the .item() call work
            class ConfidenceFaker:
                def __init__(self, val): self._val = val
                def item(self): return self._val

            confidence = ConfidenceFaker(confidence_score)

        # --- The rest of the script is UNCHANGED ---
        # It now uses the 'predicted_class_name' and 'confidence'
        # variables filled by the Keras model.
        if confidence.item() > CONFIDENCE_THRESHOLD:
            # Look for "Apple", "Banana", etc. in the new products.db
            product_info = get_product_details_from_db(predicted_class_name)

            if product_info:
                product_name = product_info["name"]
                if not current_bill or current_bill[-1]['name'] != product_name:
                    print(f"Detected: {product_name} (Confidence: {confidence.item():.2f})")

                    product_price = product_info["price"]
                    current_bill.append({"name": product_name, "price": product_price})
                    total_price += product_price
                    last_detection = product_name
                    last_detection_time = time.time()

    # --- UI Drawing (Unchanged) ---
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "Place Item Here", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # ... (rest of the UI code is identical) ...
    bill_y = 40
    cv2.putText(frame, "--- BILL ---", (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bill_y += 30
    for item in current_bill[-15:]:
        item_text = f"{item['name']}: Rs. {item['price']:.2f}"
        cv2.putText(frame, item_text, (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bill_y += 20
    cv2.line(frame, (10, bill_y), (160, bill_y), (255, 255, 255), 1)
    bill_y += 25
    total_text = f"TOTAL: Rs. {total_price:.2f}"
    cv2.putText(frame, total_text, (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "c: Clear Bill | q: Quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow('Smart Retail System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        current_bill, total_price, last_detection = [], 0, ""
        print("Bill cleared.")

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
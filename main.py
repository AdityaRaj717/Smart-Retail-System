import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import json
import time
import numpy as np
import sqlite3  # <-- Added for database
import os

# --- Main Application Logic ---

# Configuration
path_to_model = 'model/'
MODEL_PATH = os.path.join(path_to_model, 'fruit_classifier.pth')
MAPPING_PATH = os.path.join(path_to_model, 'class_mapping.json')
DATABASE_PATH = 'products.db'

# Image size must match what the model was trained on
IMG_SIZE = 224

# Confidence threshold for adding an item
# (e.g., 0.8 = 80% confident)
CONFIDENCE_THRESHOLD = 0.8

# Cooldown in seconds between adding the same item
detection_cooldown = 3.0

# --- Database Helper Function ---

def get_product_details_from_db(class_name):
    """
    Queries the products.db for a fruit's details
    based on its class_name (folder name).
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        # Row_factory allows accessing columns by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT display_name, price FROM products WHERE class_name = ?", (class_name,))
        row = cursor.fetchone()

        if row:
            return {"name": row["display_name"], "price": row["price"]}
        else:
            print(f"Warning: class_name '{class_name}' not found in database.")
            return None

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Model Loading ---

print("Loading model and class mapping...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the class mapping
try:
    with open(MAPPING_PATH, 'r') as f:
        class_to_idx = json.load(f)
except FileNotFoundError:
    print(f"Error: '{MAPPING_PATH}' not found. Please run the training script.")
    exit()

# Create the reverse mapping (index -> class name)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(idx_to_class)
print(f"Found {num_classes} classes.")

# 2. Initialize the Model (ResNet18)
# We must build the same model structure as when we trained
model = models.resnet18(pretrained=False) # No need to download weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 3. Load the trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: '{MODEL_PATH}' not found. Please run the training script.")
    exit()

model = model.to(device)
model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# --- Transforms ---
# Must match the 'test' transforms from your training script
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE), # IMG_SIZE is 224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

    # Get the region of interest
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Only process if cooldown has passed
    if time.time() - last_detection_time > detection_cooldown:
        # Preprocess ROI
        # Convert from BGR (cv2) to RGB (PIL/PyTorch)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = inference_transforms(roi_rgb).unsqueeze(0).to(device)

        # --- Get Classification ---
        with torch.no_grad():
            outputs = model(input_tensor)

            # Apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get the top class (index and confidence)
            confidence, predicted_idx = torch.max(probabilities, 0)

        # --- Check Confidence and Update Bill ---
        if confidence.item() > CONFIDENCE_THRESHOLD:
            # Convert index (e.g., 0) to class name (e.g., 'Apple_Golden_1')
            predicted_class_name = idx_to_class[predicted_idx.item()]

            # Get product details from our new database
            product_info = get_product_details_from_db(predicted_class_name)

            if product_info:
                product_name = product_info["name"]

                # Check if it's a new item (or different from the last one)
                if not current_bill or current_bill[-1]['name'] != product_name:
                    print(f"Detected: {product_name} (Confidence: {confidence.item():.2f})")

                    product_price = product_info["price"]
                    current_bill.append({"name": product_name, "price": product_price})
                    total_price += product_price
                    last_detection = product_name
                    last_detection_time = time.time()

    # --- UI Drawing ---
    # Draw the ROI box
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "Place Item Here", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Display the Bill ---
    # (This section is unchanged from your original file)
    bill_y = 40
    cv2.putText(frame, "--- BILL ---", (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bill_y += 30
    # Display up to 15 items
    for item in current_bill[-15:]:
        item_text = f"{item['name']}: Rs. {item['price']:.2f}"
        cv2.putText(frame, item_text, (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bill_y += 20

    # Draw a line and the total
    cv2.line(frame, (10, bill_y), (160, bill_y), (255, 255, 255), 1)
    bill_y += 25
    total_text = f"TOTAL: Rs. {total_price:.2f}"
    cv2.putText(frame, total_text, (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display instructions
    cv2.putText(frame, "c: Clear Bill | q: Quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show the final frame
    cv2.imshow('Smart Retail System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        current_bill, total_price, last_detection = [], 0, ""
        print("Bill cleared.")

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- Configuration ---
# Make sure these paths are correct
MODEL_DIR = "model/model.savedmodel" # Path to the SAVEDMODEL directory
LABELS_PATH = "model/labels.txt"

# !!! IMPORTANT !!!
# Set this to the path of an image you want to test
IMAGE_TO_TEST = "path/to/your/apple.jpg"
# ---

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

print("Loading model...")
# 1. LOAD THE MODEL (The Keras 3 Way)
# We use TFSMLayer, as required by the error message you got
try:
    model = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint='serving_default')
except FileNotFoundError:
    print(f"Error: Model directory not found at {MODEL_DIR}")
    print("Please make sure the 'model.savedmodel' directory is in the 'model/' folder.")
    exit()

# 2. LOAD THE LABELS (The "main.py" way, which is cleaner)
# This reads "0 Apple" and just stores "Apple"
class_names = []
try:
    with open(LABELS_PATH, "r") as f:
        for line in f:
            class_names.append(line.strip().split(' ', 1)[-1])
except FileNotFoundError:
    print(f"Error: Labels file not found at {LABELS_PATH}")
    print("Please make sure 'labels.txt' is in the 'model/' folder.")
    exit()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load and process the test image
try:
    image = Image.open(IMAGE_TO_TEST).convert("RGB")
except FileNotFoundError:
    print(f"---")
    print(f"Error: Test image not found at '{IMAGE_TO_TEST}'")
    print("Please update the 'IMAGE_TO_TEST' variable in this script to a real image path.")
    print(f"---")
    exit()

# resizing the image (from your sample code)
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image (from your sample code)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# 3. PREDICT (The Keras 3 Way)
# The model is called like a function, not with .predict()
# It returns a DICTIONARY, e.g., {'output_layer_name': tensor}
prediction_dict = model(data)

# 4. GET THE RESULTS (The Fix for 'KeyError: 0')
# We must get the tensor from the dictionary's values
# This was the fix that your 'main.py' script was missing
prediction_tensor = list(prediction_dict.values())[0]

# Now we can use this tensor just like your old code
index = np.argmax(prediction_tensor)

# Get the class name from our cleaned list
class_name = class_names[index]

# Get the confidence score
confidence_score = prediction_tensor[0][index]

# Print prediction and confidence score
print("\n--- PREDICTION ---")
print(f"Class: {class_name}")
print(f"Confidence Score: {confidence_score}")
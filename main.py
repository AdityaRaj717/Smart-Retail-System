import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import json
import time
import numpy as np

# --- NEW: Squeeze-and-Excitation Block ---
# This block helps the model focus on the most important features.
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- UPDATED: ProductCNN_v2 with Attention ---
class ProductCNN_v2(nn.Module):
    def __init__(self, num_classes):
        super(ProductCNN_v2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(32), # <-- Attention Block Added

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(64), # <-- Attention Block Added

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(128), # <-- Attention Block Added
        )
        # For an input of 150x150, the flattened size is 128 * 18 * 18
        self.classifier = nn.Sequential(
            nn.Linear(128 * 18 * 18, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- ADD THIS TO YOUR main.py SCRIPT ---

# This model is based on ProductCNN_v2 but outputs a feature vector (embedding)
# instead of classification scores.
class ProductEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ProductEmbeddingModel, self).__init__()
        # We use the same powerful feature extractor with attention
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(128),
        )
        # The new "head" of the model outputs the embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(128 * 18 * 18, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim) # Output a 128-dimensional vector
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_head(x)
        # L2-normalize the embedding
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


path_to_model = 'model/'

print("CUDA Avilable:" ,torch.cuda.is_available())


# Maps the integer class index from your model to the product's name and price.
product_db = {
    0: {"name": "Butter cookies", "price": np.random.randint(80, 120)},
    1: {"name": "Chana Chur", "price": np.random.randint(50, 90)},
    2: {"name": "Chipotle sauce", "price": np.random.randint(150, 250)},
    3: {"name": "Punjabi tadka", "price": np.random.randint(40, 70)},
    4: {"name": "Shahi Dates", "price": np.random.randint(200, 300)},
    5: {"name": "Spicy Coated Peanuts", "price": np.random.randint(60, 100)}
}
print("Product Database:", product_db)



class ProductCNN(nn.Module):
    def __init__(self, num_classes):
        super(ProductCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Using a fixed size from training, adjust if your IMG_SIZE was different
        # For an input of 150x150, the flattened size is 128 * 18 * 18
        self.classifier = nn.Sequential(
            nn.Linear(128 * 18 * 18, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the class mapping
try:
    with open(path_to_model + 'class_mapping.json', 'r') as f:
        idx_to_class = json.load(f)
        # JSON saves keys as strings, so convert them back to integers
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)
except FileNotFoundError:
    print("Error: 'class_mapping.json' not found. Please run the training script first.")
    exit()

device = torch.device("cpu")

# Instantiate the model and load the trained weights
model = ProductCNN_v2(num_classes).to(device)
try:
    model.load_state_dict(torch.load(path_to_model + 'smart_retail_model.pth', map_location=device))
except FileNotFoundError:
    print("Error: 'smart_retail_model.pth' not found. Please run the training script first.")
    exit()

model.eval()

IMG_SIZE = 150
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Warming up the camera...")
time.sleep(2.0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ROI (Region of Interest) coordinates - a box in the center of the frame
roi_x, roi_y, roi_w, roi_h = 170, 90, 300, 300

# Billing and detection logic variables
current_bill = []
total_price = 0
last_detection = ""
detection_cooldown = 3.0
last_detection_time = time.time()
CONFIDENCE_THRESHOLD = 0.90


print("Starting live billing system... Press 'q' to quit, 'c' to clear bill.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Check if enough time has passed since the last detection
    if time.time() - last_detection_time > detection_cooldown:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = inference_transforms(roi_rgb).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # If confidence is high, add item to the bill
        if confidence.item() > CONFIDENCE_THRESHOLD:
            # Get the integer index directly from the model's prediction
            predicted_index = predicted_idx.item()

            # Look up the product info using the index
            product_info = product_db.get(predicted_index)

            # Check if the product exists in our database
            if product_info:
                product_name = product_info["name"]
                product_price = product_info["price"]

                # Add to bill only if it's a different item from the last one
                if not current_bill or current_bill[-1]['name'] != product_name:
                    print(f"Detected: {product_name} with {confidence.item():.2f} confidence")

                    # Append the details to our bill
                    current_bill.append({"name": product_name, "price": product_price})
                    total_price += product_price
                    last_detection = product_name
                    last_detection_time = time.time() # Reset cooldown timer

    # Draw the ROI box
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "Place Item Here", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the last detected item
    if last_detection:
        cv2.putText(frame, f"Last Scan: {last_detection}", (roi_x, roi_y + roi_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Display the Bill ---
    bill_y = 40
    cv2.putText(frame, "--- BILL ---", (10, bill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bill_y += 30
    for item in current_bill:
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
    if key == ord('q'):
        break
    if key == ord('c'):
        # Clear the bill
        current_bill = []
        total_price = 0
        last_detection = ""
        print("Bill cleared.")

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
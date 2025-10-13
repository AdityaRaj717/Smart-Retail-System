import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import json
import time
import numpy as np
import faiss

# --- Squeeze-and-Excitation Block ---
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

# --- Product Embedding Model ---
class ProductEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ProductEmbeddingModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), SEBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), SEBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), SEBlock(128),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(128 * 18 * 18, 512), nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_head(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# --- Main Application Logic ---

# Configuration
path_to_model = 'model/'
EMBEDDING_DIM = 128
IMG_SIZE = 150
DISTANCE_THRESHOLD = 0.8 # Similarity threshold for adding an item

# Load the product database information (names, prices)
# Old db
product_db = {
    0: {"name": "Butter cookies", "price": 35},
    1: {"name": "Chana Chur", "price": 110},
    2: {"name": "Chipotle sauce", "price": 73},
    3: {"name": "Punjabi tadka", "price": 55},
    4: {"name": "Shahi Dates", "price": 380},
    5: {"name": "Spicy Coated Peanuts", "price": 270}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProductEmbeddingModel(embedding_dim=EMBEDDING_DIM).to(device)
try:
    model.load_state_dict(torch.load(path_to_model + 'smart_retail_embedding_model.pth', map_location=device))
except FileNotFoundError:
    print("Error: 'smart_retail_embedding_model.pth' not found. Please run the training script for the embedding model.")
    exit()
model.eval()

# Load the Faiss index and product mapping
try:
    faiss_index = faiss.read_index(path_to_model + "product_index.faiss")
    with open(path_to_model + 'index_to_product_id.json', 'r') as f:
        index_to_product_id = json.load(f)
        index_to_product_id = {int(k): v for k, v in index_to_product_id.items()}

except FileNotFoundError:
    print("Error: Faiss index or product mapping not found. Please run 'create_database.py' first.")
    exit()

# Image transforms
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Camera and Billing Setup ---
cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_w, roi_h = 170, 90, 300, 300
current_bill = []
total_price = 0
last_detection = ""
detection_cooldown = 3.0
last_detection_time = time.time()

print("Starting live billing system... Press 'q' to quit, 'c' to clear bill.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    if time.time() - last_detection_time > detection_cooldown:
        # Preprocess ROI
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = inference_transforms(roi_rgb).unsqueeze(0).to(device)

        # --- Get Embedding and Search ---
        with torch.no_grad():
            embedding = model(input_tensor).cpu().numpy()
            # Search the faiss index for the nearest neighbor
            D, I = faiss_index.search(embedding, 1) # D=distance, I=index
            distance = D[0][0]
            nearest_index = I[0][0]

        # If the match is close enough, add the item to the bill
        if distance < DISTANCE_THRESHOLD:
            product_id = index_to_product_id[nearest_index]
            product_info = product_db.get(product_id)

            if product_info:
                product_name = product_info["name"]
                if not current_bill or current_bill[-1]['name'] != product_name:
                    print(f"Detected: {product_name} (Distance: {distance:.2f})")
                    product_price = product_info["price"]
                    current_bill.append({"name": product_name, "price": product_price})
                    total_price += product_price
                    last_detection = product_name
                    last_detection_time = time.time()

    # --- UI Drawing ---
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.imshow('Smart Retail System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        current_bill, total_price, last_detection = [], 0, ""
        print("Bill cleared.")

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
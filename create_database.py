import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import json
import os

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

DATA_DIR = 'processed_data/train'
MODEL_PATH = 'model/smart_retail_embedding_model.pth'
INDEX_SAVE_PATH = 'model/product_index.faiss'
MAP_SAVE_PATH = 'model/index_to_product_id.json'
EMBEDDING_DIM = 128
IMG_SIZE = 150

inference_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProductEmbeddingModel(embedding_dim=EMBEDDING_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Generate Embeddings ---
print("Generating embeddings for all products...")
all_embeddings = []
index_to_product_id = {}
current_index = 0
product_id_counter = 0

# Create a mapping from class name to a numeric ID
class_to_id = {name: i for i, name in enumerate(os.listdir(DATA_DIR))}


for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    product_id = class_to_id[class_name]

    # We've used just one representative image per class for the database
    image_file = os.listdir(class_path)[0]
    image_path = os.path.join(class_path, image_file)

    image = Image.open(image_path).convert('RGB')
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy()
        all_embeddings.append(embedding.flatten())
        index_to_product_id[current_index] = product_id
        current_index += 1

# --- Create and Save Faiss Index ---
embeddings_matrix = np.array(all_embeddings).astype('float32')
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(embeddings_matrix)

print(f"Database created with {index.ntotal} product embeddings.")

faiss.write_index(index, INDEX_SAVE_PATH)
with open(MAP_SAVE_PATH, 'w') as f:
    json.dump(index_to_product_id, f)

print(f"Faiss index saved to {INDEX_SAVE_PATH}")
print(f"Product ID mapping saved to {MAP_SAVE_PATH}")
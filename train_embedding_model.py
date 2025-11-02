import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

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

# --- Dataset for Triplet Loss ---
class TripletProductDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.class_to_images = self._get_class_to_images()
        self.classes = list(self.class_to_images.keys())
        self.image_list = self._get_image_list()

    def _get_class_to_images(self):
        class_to_images = {}
        for class_name in os.listdir(self.image_folder):
            class_path = os.path.join(self.image_folder, class_name)
            if os.path.isdir(class_path):
                class_to_images[class_name] = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        return class_to_images

    def _get_image_list(self):
        image_list = []
        for class_name, images in self.class_to_images.items():
            for image_path in images:
                image_list.append((class_name, image_path))
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        anchor_class, anchor_path = self.image_list[index]

        # Get a positive image (same class, different image)
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.class_to_images[anchor_class])

        # Get a negative image (different class)
        negative_class = anchor_class
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)
        negative_path = random.choice(self.class_to_images[negative_class])

        # Load and transform images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

DATA_DIR = 'processed_data/train'
MODEL_SAVE_PATH = 'model/smart_retail_embedding_model.pth'
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
IMG_SIZE = 150

# Data transformations
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Setup dataset and dataloader
dataset = TripletProductDataset(image_folder=DATA_DIR, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Setup model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProductEmbeddingModel(embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training for the embedding model...")

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

print("Finished Training.")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm
import faiss
import json
import os

# --- CHANGED: Define paths to your new dataset ---
# !!! IMPORTANT: Update these paths to where you downloaded Fruits 360 !!!
TRAIN_DATA_DIR = '/path/to/your/fruits-360/Training'
TEST_DATA_DIR = '/path/to/your/fruits-360/Test'

# --- Configuration ---
BATCH_SIZE = 64 # Increased batch size for faster training
EMBEDDING_DIM = 512  # This is fixed by ResNet18's fc layer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5 # 5-10 epochs is usually enough for fine-tuning

# --- CHANGED: Added separate transforms for Training (with augmentation) and Testing ---

# 1. Augmentations for the Training set
# This "messes up" the perfect images to make them look like your webcam feed
train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),     # Take a random 224x224 crop

    # --- This is the "secret sauce" to bridge the domain gap ---
    T.RandomRotation(degrees=30),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    # --- End of augmentations ---

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Simple transforms for the Test set (to get metrics)
# We don't augment the test set, just resize and normalize
test_transforms = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224), # Use a standard center crop
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(EmbeddingNet, self).__init__()
        # Load pre-trained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        return self.model(x)

def train_embedding_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)

        # Note: We are not using labels here, this is unsupervised on embeddings
        # This seems to be the original intent of the code.
        # If this was a classification task, we'd calculate loss against labels.
        # For a pure embedding model, the "training" is just running it.
        # Let's assume this is a placeholder and we're just saving the model.
        # *** RE-IMPLEMENTING AS A CLASSIFICATION TASK TO GET METRICS ***
        # The previous code had no loss function. We MUST train it as a classifier
        # to get a meaningful model and metrics.

        # --- CHANGED: Re-implemented training as a classification task ---
        # We need a new model definition for this
        pass # See new model definition below

    # --- The original script was not training the model, just passing data through. ---
    # --- Let's replace the model and training loop to do it properly. ---

class ClassificationNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM):
        super(ClassificationNet, self).__init__()
        # Load pre-trained ResNet18
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features

        # We'll keep the embedding layer
        self.base_model.fc = nn.Linear(num_ftrs, embedding_dim)

        # And add a new classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embeddings = self.base_model(x)
        output = self.classifier(embeddings)
        return output

    def get_embedding(self, x):
        return self.base_model(x)

def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) # Get classification outputs
        loss = criterion(outputs, labels) # Calculate loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# --- ADDED: A function to get your metrics! ---
def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    product_ids = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Getting Embeddings"):
            images = images.to(device)
            emb = model.get_embedding(images) # Use the specific embedding getter
            embeddings.append(emb.cpu().numpy())
            product_ids.extend(labels.cpu().numpy())
    return np.vstack(embeddings), np.array(product_ids)

def main():
    # --- CHANGED: Load both train and test datasets ---
    train_dataset = ImageFolder(TRAIN_DATA_DIR, transform=train_transforms)
    test_dataset = ImageFolder(TEST_DATA_DIR, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")

    # --- CHANGED: Save class mapping from the train dataset ---
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open('model/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
    print("Class mapping saved.")

    num_classes = len(class_mapping)

    # --- CHANGED: Use the new ClassificationNet ---
    model = ClassificationNet(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- CHANGED: Added a full training and testing loop ---
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")

        # Train
        train_loss = train_loop(model, train_loader, criterion, optimizer, DEVICE)

        # Test (This gets your metrics!)
        test_accuracy = test_model(model, test_loader, DEVICE)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    print("Training complete. Saving model...")
    # Save the trained model
    torch.save(model.state_dict(), 'model/smart_retail_embedding_model.pth')

    # --- Build FAISS index ---
    print("Building FAISS index...")
    # We get embeddings from the training set to build the search index
    embeddings, product_ids = get_embeddings(model, train_loader, DEVICE)

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIDMap(index)

    # We map the FAISS index ID to the product_id (which is the class label)
    index_to_product_id = {i: int(pid) for i, pid in enumerate(product_ids)}

    index.add_with_ids(embeddings, np.arange(len(product_ids)))

    print("Saving FAISS index and mappings...")
    faiss.write_index(index, 'model/product_index.faiss')
    with open('model/index_to_product_id.json', 'w') as f:
        json.dump(index_to_product_id, f)

    print("All done!")

if __name__ == '__main__':
    # Ensure 'model' directory exists
    os.makedirs('model', exist_ok=True)
    main()
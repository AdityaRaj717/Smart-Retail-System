import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import json
import time
import copy

# --- CONFIGURATION ---

# 1. IMPORTANT: Set this variable to the parent directory
#    that CONTAINS your 'Training' and 'Test' folders.
#
#    Examples:
#    DATASET_PATH = 'C:/Users/YourUser/Desktop/fruits-360'
#    DATASET_PATH = '/home/youruser/datasets/fruits-360'
#
DATASET_PATH = '/mnt/Personal/Programming/Self Learning/Projects/Capstone/Smart Retail Checkout System/dataset/fruits-360_100x100/fruits-360'  # <-- !!! UPDATE THIS LINE !!!

# 2. Output directory for the new model and mapping file.
#    This script will create the 'model/' folder if it doesn't exist.
MODEL_SAVE_DIR = 'model'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'fruit_classifier.pth')
MAPPING_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'class_mapping.json')

# 3. Training hyperparameters (you can tune these)
BATCH_SIZE = 32
NUM_EPOCHS = 10  # 10 epochs is a good start with a pre-trained model
LEARNING_RATE = 0.001

# --- END CONFIGURATION ---


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    """
    Main training and validation loop.
    Saves the best performing model based on validation accuracy.
    """
    start_time = time.time()

    # We'll keep track of the best model's weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and a validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # Adding a simple progress bar
            num_batches = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Print progress
                if (i+1) % 100 == 0 or (i+1) == num_batches:
                    print(f'\r  [{phase}] Batch {i+1}/{num_batches}...', end='')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(best_model_wts, MODEL_SAVE_PATH)
                print(f"  New best model saved to {MODEL_SAVE_PATH} (Acc: {best_acc:.4f})")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Acc: {best_acc:.4f}')

    # Load best model weights back
    model.load_state_dict(best_model_wts)
    return model

# --- Main execution ---
if __name__ == "__main__":
    # 1. Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define data transformations
    # ResNet models expect 224x224 images
    # Use ImageNet stats for normalization as we use a pretrained model
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }

    # 3. Load Datasets and create DataLoaders
    print("Loading datasets...")
    train_data_path = os.path.join(DATASET_PATH, 'Training')
    test_data_path = os.path.join(DATASET_PATH, 'Test')

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"--- ERROR ---")
        print(f"'Training' or 'Test' folder not found in: '{DATASET_PATH}'")
        print("Please update the 'DATASET_PATH' variable in this script.")
        exit()

    image_datasets = {
        'train': datasets.ImageFolder(train_data_path, data_transforms['train']),
        'test': datasets.ImageFolder(test_data_path, data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes (fruit types).")

    # 4. Save the class mapping file
    # This is CRITICAL for main.py to understand the model's output
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    class_mapping = image_datasets['train'].class_to_idx
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump(class_mapping, f)
    print(f"Saved class mapping to {MAPPING_SAVE_PATH}")

    # 5. Initialize the Model
    print("Loading pre-trained ResNet18 model...")
    # We use a model pretrained on ImageNet
    model = models.resnet18(pretrained=True)

    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features

    # Replace the final layer to match our number of fruit classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Send the model to the configured device (GPU or CPU)
    model = model.to(device)

    # 6. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()

    # We'll only optimize the parameters of the final layer we just added
    # For a more advanced approach, you could fine-tune all parameters
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Start Training
    print("Starting training...")
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)

    print("\n--- SUCCESS ---")
    print(f"Training finished. Best model saved to: {MODEL_SAVE_PATH}")
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mobilenetv4 import mobilenetv4_conv_small_grayscale  # Your modified model
from music.globals import MODELS_DIR, SPECTROGRAM_LABELS_DIR


# --- Step 1: Load all JSON metadata ---
def load_dataset(root_dir="data"):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


# --- Load Model ---
def load_model(model_name="models/mobilenetv4_dance_classifier.pth", num_classes=10, models_dir=MODELS_DIR):
    model = mobilenetv4_conv_small_grayscale(num_classes=num_classes)
    model.load_state_dict(torch.load(MODELS_DIR + "/" + model_name, map_location=torch.device('cpu')))
    model.eval()
    return model


# --- Custom Dataset Class with Spectrogram Augmentation ---
class DanceDataset(Dataset):
    def __init__(self, json_files, dance_to_idx):
        self.json_files = json_files
        self.dance_to_idx = dance_to_idx

    def __len__(self):
        return len(self.json_files)

    def augment_spectrogram(self, spectrogram):
        """Apply random augmentations to spectrogram"""
        # Add Gaussian noise
        if np.random.rand() < 0.5:  # 50% chance
            noise = np.random.normal(0, 0.05, spectrogram.shape)  # Scale to dB range
            spectrogram += noise
            spectrogram = np.clip(spectrogram, 0, 1)  # Clamp to [0, 1] dB

        # Time masking (random time segment zeroed)
        if np.random.rand() < 0.2:  # 20% chance
            t_start = np.random.randint(0, spectrogram.shape[1] - 10)
            spectrogram[:, t_start:t_start + 10] = 0  # Mask 10 time steps

        # Frequency masking (random frequency band zeroed)
        if np.random.rand() < 0.3:  # 30% chance
            f_start = np.random.randint(0, spectrogram.shape[0] - 5)
            spectrogram[f_start:f_start + 5, :] = 0  # Mask 5 frequency bins

        return spectrogram

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Load spectrogram from .npy file
        spectrogram_path = metadata["spectrogram_path"]
        spectrogram = np.load(spectrogram_path).astype(np.float32)  # FIXME: How fast is your disk ?
        # Model size is about 10 Mb, but I have 6+ Gb dataset with 30k files

        # Apply augmentations
        spectrogram = self.augment_spectrogram(spectrogram)

        # Convert to tensor [1, 128, T]
        spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).float()

        # Map dance label to integer
        label = self.dance_to_idx[metadata["dance"]]

        return spectrogram_tensor, label


# --- Create Dataset and DataLoader ---
def create_data_loaders(root_dir="data", batch_size=64, val_split=0.1):
    # Step 1: Load all JSON files
    json_files = load_dataset(root_dir)

    # Step 2: Create dance-to-index mapping
    dance_labels = [json.load(open(f))["dance"] for f in json_files]
    unique_dances = sorted(set(dance_labels))
    dance_to_idx = {dance: idx for idx, dance in enumerate(unique_dances)}
    print(f"Found {len(unique_dances)} dance types: {unique_dances}")

    # Step 3: Create dataset
    dataset = DanceDataset(json_files, dance_to_idx)

    # Step 4: Split into train/val
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )  # TODO: file system based dataset split

    # Step 5: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(unique_dances)


# --- Step 4: Training Loop ---
def train_model(model, train_loader, val_loader, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu",
                model_save_name=None, models_dir=MODELS_DIR, lr=1e-4):
    print("Using device {}".format(device))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a directory for saving models
    os.makedirs(models_dir, exist_ok=True)

    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if model_save_path is not None and (epoch == 0 or val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(model.state_dict(), models_dir + "/" + model_save_name)
            print(f"Saved best model at epoch {epoch + 1} with {val_acc:.2f}% accuracy")


# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Create DataLoaders
    train_loader, val_loader, num_classes = create_data_loaders(root_dir=SPECTROGRAM_LABELS_DIR, batch_size=64)

    model_load_path = "mobilenetv4_dance_classifier_noised.pth"
    model_save_path = "mobilenetv4_dance_classifier_noised_v2.pth"

    # Step 2: Initialize Model
    model = None
    if model_load_path is None:
        model = mobilenetv4_conv_small_grayscale(num_classes=num_classes)
    else:
        model = load_model(model_load_path, num_classes=num_classes)

    # Step 3: Train
    train_model(model, train_loader, val_loader, num_epochs=10, model_save_name=model_save_path)

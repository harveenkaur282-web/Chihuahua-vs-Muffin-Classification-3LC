import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ResNet18Classifier
from src.data import get_train_transforms, get_val_transforms

# Config
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
RANDOM_SEED = 42
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
NUM_CLASSES = 2
BEST_MODEL_FILENAME = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()
    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }

def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent.parent
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    def train_fn(sample):
        image = Image.open(sample["image"]).convert("RGB")
        return train_transform(image), sample["label"]

    def val_fn(sample):
        image = Image.open(sample["image"]).convert("RGB")
        return val_transform(image), sample["label"]

    train_table = tlc.Table.from_names(PROJECT_NAME, DATASET_NAME, "train").latest()
    val_table = tlc.Table.from_names(PROJECT_NAME, DATASET_NAME, "val").latest()

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)
    
    train_sampler = train_table.create_sampler(exclude_zero_weights=True)
    train_loader = DataLoader(train_table, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_table, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    run = tlc.init(PROJECT_NAME, description="Training Chihuahua vs Muffin")
    
    best_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        print(f"Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), base_path / BEST_MODEL_FILENAME)
        
        tlc.log({"epoch": epoch, "val_accuracy": val_acc})
        scheduler.step()

    run.set_status_completed()
    print(f"Done. Best Accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train()

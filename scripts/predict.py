import torch
import csv
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ResNet18Classifier
from src.data import TestDataset, get_val_transforms

# Config
MODEL_PATH = "models/best_model.pth"
TEST_DIR = "data/test"
OUTPUT_PATH = "outputs/submission.csv"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    base_path = Path(__file__).parent.parent
    model_path = base_path / MODEL_PATH
    test_dir = base_path / TEST_DIR
    output_path = base_path / OUTPUT_PATH

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model = ResNet18Classifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = TestDataset(test_dir, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
            
            for i, p, c in zip(ids, pred.cpu().numpy(), conf.cpu().numpy()):
                predictions.append({"image_id": i, "prediction": int(p), "confidence": float(c)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()

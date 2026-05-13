import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Standard ImageNet normalization
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_train_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        NORMALIZE,
    ])

def get_val_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        NORMALIZE,
    ])

class TestDataset(Dataset):
    """
    Dataset for unlabeled test images.
    Returns (image, image_id).
    """
    def __init__(self, image_dir, transform=None, image_size=128):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        self.images = []
        
        if self.image_dir.exists():
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in self.image_dir.glob(ext):
                    if img.name.lower() not in seen:
                        seen.add(img.name.lower())
                        self.images.append(img)
        self.images.sort(key=lambda x: x.name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
            
        if self.transform:
            image = self.transform(image)
        
        return image, img_path.stem

import os
import json

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, image_folder, description_file, transform=None):

        self.image_folder = image_folder
        self.transform = transform

        # Load descriptions
        with open(description_file, "r") as f:
            self.descriptions = json.load(f)

        # Get image names
        self.image_names = list(self.descriptions.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # Load image with error handling
        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            raise ValueError(f"Error loading image {image_name}: {e}")

        if self.transform:
            image = self.transform(image)

        # Get corresponding description
        description = self.descriptions[image_name]
        
        return image, description

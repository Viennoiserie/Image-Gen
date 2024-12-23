import os
import json
import torch
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

        image = self.add_alpha_channel(image)

        if self.transform:
            image = self.apply_transformations(image)

        # Get corresponding description
        description = self.descriptions[image_name]
        
        return image, description

    def add_alpha_channel(self, image):

        """ Add a dummy alpha channel to an image to make it RGBA """

        image_tensor = transforms.ToTensor()(image)  

        alpha_channel = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))  
        image_with_alpha = torch.cat([image_tensor, alpha_channel], dim=0)  

        return image_with_alpha

    def apply_transformations(self, image):

        """ Apply transformations while handling the 4th alpha channel """

        # Split the RGB and alpha channels
        rgb_image = image[:3]  

        # Convert the RGB part to a PIL Image and apply transformations
        rgb_image_pil = transforms.ToPILImage()(rgb_image)

        # Convert the transformed RGB back to a tensor
        rgb_tensor = transforms.ToTensor()(rgb_image_pil)

        # Reattach the original alpha channel (no changes made)
        alpha_channel = image[3:]  

        # Concatenate the transformed RGB tensor with the alpha channel
        image_with_alpha = torch.cat([rgb_tensor, alpha_channel], dim=0)  

        return image_with_alpha


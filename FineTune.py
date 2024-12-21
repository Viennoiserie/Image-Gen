# region : Imports

import torch

from torch.optim import AdamW
from torch.amp import GradScaler 
from huggingface_hub import login
from torchvision import transforms
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

# endregion

# region : Model Setup

# Authentication
login("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")

# Load pre-trained Stable Diffusion pipeline
model_name = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Load components for fine-tuning
text_encoder = CLIPTextModel.from_pretrained(model_name).to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_name).to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(model_name)

# endregion

# region : Variables Setup

# File paths
image_folder = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\images"
description_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\labels\unified_descriptions.json"

# Transformations for images
transform = transforms.Compose([

    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5])  # Normalize for Stable Diffusion
])

# Load dataset and create DataLoader
dataset = ImageDataset(image_folder, description_file, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 for VRAM efficiency

# endregion

# region : Training Preparation

# Optimizer
optimizer = AdamW(unet.parameters(), lr=1e-5, weight_decay=0.01)

# Scheduler
num_training_steps = len(train_dataloader) * 10  # Adjust for number of epochs

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,  
    num_training_steps=num_training_steps
)

# Mixed Precision Training
scaler = GradScaler()

# Gradient accumulation
gradient_accumulation_steps = 4
accumulated_loss = 0.0

# endregion

# region : Training

# Training loop
num_epochs = 10  # Increase epochs for better results

for epoch in range(num_epochs):

    print(f"Starting epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(train_dataloader):

        images, descriptions = batch

        # Tokenize text prompts
        inputs = tokenizer(descriptions, padding="max_length", max_length=77, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")

        # Move images to GPU
        pixel_values = images.to("cuda")

        # Forward pass with mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = unet(pixel_values, input_ids=input_ids)
            loss = outputs.loss  

        # Backward pass with gradient accumulation
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler step
            lr_scheduler.step()

            # Log progress
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {accumulated_loss / gradient_accumulation_steps:.4f}")
            accumulated_loss = 0.0

    # Save intermediate model checkpoints
    pipeline.save_pretrained(f"fine_tuned_model_epoch_{epoch + 1}")

# endregion

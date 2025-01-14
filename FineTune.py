# region : Imports

import os
import gc
import torch

from torch.optim import AdamW
from torch.amp import GradScaler
from huggingface_hub import login
from torchvision import transforms
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler

# endregion

# region : Setup

# Avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Authentication
login("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")  # Replace with your Hugging Face token

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# endregion

# region : Model

model_name = "runwayml/stable-diffusion-v1-5"

try:
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

except Exception as e:
    print(f"Error loading pipeline: {e}")

# Load U-Net and the text encoder
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)

# Scheduler for training
noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

# endregion

# region : Dataset Preparation

# File paths
image_folder = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\images"
description_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\labels\unified_descriptions.json"

# Transformations for images
transform = transforms.Compose([

    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Load dataset and create DataLoader
dataset = ImageDataset(image_folder, description_file, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# endregion

# region : Optimizer and Scheduler

optimizer = AdamW(unet.parameters(), lr=5e-6, weight_decay=0.01)

num_training_steps = len(train_dataloader) * 10  

lr_scheduler = get_scheduler("cosine",
                             optimizer=optimizer,
                             num_warmup_steps=1000,
                             num_training_steps=num_training_steps)

# Mixed Precision Training
scaler = GradScaler()

# Gradient accumulation
gradient_accumulation_steps = 4
accumulated_loss = 0.0

# endregion

# region : Training Function

def train_unet():

    num_epochs = 10
    max_mem_alloc = 4  

    for epoch in range(num_epochs):

        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(train_dataloader):

            images, descriptions = batch

            # Tokenize text descriptions
            inputs = tokenizer(descriptions,
                                
                               padding="max_length",

                               truncation=True, 
                               max_length=77,

                               return_tensors="pt")
            
            input_ids = inputs.input_ids.to(device)

            # Text embeddings
            text_embeddings = text_encoder(input_ids).last_hidden_state

            # Prepare image batch 
            pixel_values = images.to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=device).long()

            # Add noise to images
            noise = torch.randn_like(pixel_values)
            noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

            # Forward pass with autocast for mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                model_pred = unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):

                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                lr_scheduler.step()

                print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {accumulated_loss / gradient_accumulation_steps:.4f}")
                accumulated_loss = 0.0

            # Clear GPU memory periodically
            torch.cuda.empty_cache()

        # Garbage collection after each epoch
        gc.collect()

        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9} GB")

        # Save intermediate model checkpoints
        checkpoint_path = rf"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Models\fine_tuned_model_epoch_{epoch + 1}"
        pipeline.save_pretrained(checkpoint_path)

# endregion

# region : Main

if __name__ == "__main__":
    train_unet()

# endregion

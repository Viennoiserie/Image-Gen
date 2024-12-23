# region : Imports

import os
import gc
import torch
import warnings
import torch.utils.checkpoint

from torch.optim import AdamW
from torch.amp import GradScaler
from huggingface_hub import login
from torchvision import transforms
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

# endregion

# region : Setup

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*diffusion_pytorch_model.safetensors.*")

# Avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Authentication
login("KEY")

# endregion

# region : Model

# Load pre-trained Stable Diffusion pipeline (using a smaller model if needed)
model_name = "CompVis/stable-diffusion-v1-2"

try:
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

except Exception as e:
    print(f"Error loading pipeline: {e}")

# Load components for fine-tuning
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-2", subfolder="unet").to("cuda")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# endregion

# region : Preparation

# File paths
image_folder = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\images"
description_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\labels\unified_descriptions.json"

# Transformations for images
transform = transforms.Compose([

    transforms.Resize((64, 64)),  
    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5])  # Normalize for Stable Diffusion
])

# Load dataset and create DataLoader
dataset = ImageDataset(image_folder, description_file, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizer
optimizer = AdamW(unet.parameters(), lr=1e-5, weight_decay=0.01)

# Scheduler
num_training_steps = len(train_dataloader) * 10  

lr_scheduler = get_scheduler(

    "cosine",

    optimizer=optimizer,
    num_warmup_steps=500,

    num_training_steps=num_training_steps
)

# Mixed Precision Training
scaler = GradScaler()

# Gradient accumulation
gradient_accumulation_steps = 2  
accumulated_loss = 0.0

# endregion

# region : Training

def forward_function(pixel_values, timesteps, text_embeddings):
    return unet(sample=pixel_values, timestep=timesteps, encoder_hidden_states=text_embeddings)

def adjust_batch_size(train_dataloader, max_mem_alloc):
    
    # Check if the current memory allocation exceeds the limit
    allocated_mem = torch.cuda.memory_allocated() / 1e9  

    if allocated_mem > max_mem_alloc:
        
        new_batch_size = max(1, train_dataloader.batch_size // 2)
        
        # Recreate the DataLoader with the new batch size
        new_dataloader = DataLoader(dataset, batch_size=new_batch_size, shuffle=True)
        return new_dataloader

    return train_dataloader

num_epochs = 1
max_mem_alloc = 4 

for epoch in range(num_epochs):

    print(f"Starting epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(train_dataloader):

        # Adjust batch size if necessary based on available memory
        train_dataloader = adjust_batch_size(train_dataloader, max_mem_alloc)

        images, descriptions = batch

        inputs = tokenizer(descriptions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")

        text_embeddings = text_encoder(input_ids).last_hidden_state
        pixel_values = images.to("cuda")

        timesteps = torch.randint(0, 500, (pixel_values.shape[0],), device="cuda").long()

        with torch.autocast(device_type='cuda', dtype=torch.float16):

            try:
                # Forward pass with checkpointing
                outputs = torch.utils.checkpoint.checkpoint(forward_function, pixel_values, timesteps, text_embeddings)
                loss = outputs.loss

            except RuntimeError as e:
                print(f"Error during checkpointing: {e}")
                continue

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

    # Run garbage collection after each epoch
    gc.collect()

    # Print memory summary
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9} GB")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Save intermediate model checkpoints
    checkpoint_path = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Models\fine_tuned_model_epoch_{epoch + 1}"
    pipeline.save_pretrained(checkpoint_path)

# endregion

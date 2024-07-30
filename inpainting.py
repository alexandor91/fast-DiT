import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load the CLIP model and processor
clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# Load the images
init_image_path = "/path/to/your/warped_image.png"
mask_image_path = "/path/to/your/mask_image.png"
init_image = Image.open(init_image_path).convert("RGB")
mask_image = Image.open(mask_image_path).convert("L")

# Prepare the condition image for CLIP embedding
condition_image_path = "/path/to/your/condition_image.png"
condition_image = Image.open(condition_image_path).convert("RGB")

# Generate CLIP embedding
inputs = clip_processor(images=condition_image, return_tensors="pt")
clip_embedding = clip_model.get_image_features(**inputs)

# Ensure the images are in the correct format
init_image = np.array(init_image).astype(np.float32) / 255.0
mask_image = np.array(mask_image).astype(np.float32) / 255.0

# Convert to tensors
init_image = torch.tensor(init_image).permute(2, 0, 1).unsqueeze(0).to("cuda")
mask_image = torch.tensor(mask_image).unsqueeze(0).unsqueeze(0).to("cuda")

# Perform inpainting
with torch.no_grad():
    result = pipe(prompt=None, 
                  init_image=init_image, 
                  mask_image=mask_image, 
                  clip_embedding=clip_embedding,
                  num_inference_steps=50).images[0]

# Save the result
result.save("inpainted_image.png")
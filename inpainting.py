import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import PaintByExamplePipeline
from transformers import CLIPModel, CLIPProcessor


if __name__ == "__main__":
    # Load the Paint by Example model
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/realestate/1'
    src_type = 'rgb'
    file_type = 'warped-output'
    output_folder = 'inpaint-output'
    json_file = os.path.join(base_dir, folder_type, 'pose.json')

    model_id = "Fantasy-Studio/Paint-by-Example"
    pipe = PaintByExamplePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Load the CLIP model and processor
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    # Load the JSON file
    with open(json_file, 'r') as f:
        poses_data = json.load(f)

    # Load the images
    init_image_path = os.path.join(base_dir, folder_type, file_type, '295728767.png')
    init_image = Image.open(init_image_path).convert("RGB")
    init_image_np = np.array(init_image)

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(init_image_np, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Convert the binary mask to have values of 0 and 1
    binary_mask = (binary_mask / 255).astype(np.uint8)

    # Prepare the condition image for CLIP embedding
    condition_image_path = os.path.join(base_dir, folder_type, src_type, '295728767.png')
    condition_image = Image.open(condition_image_path).convert("RGB")

    # Generate CLIP embedding
    inputs = clip_processor(images=condition_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        clip_embedding = clip_model.get_image_features(**inputs)

    # Ensure the images are in the correct format
    init_image = init_image_np.astype(np.float32) / 255.0
    mask_image = binary_mask.astype(np.float32)

    # Convert numpy arrays back to PIL images
    init_image_pil = Image.fromarray((init_image * 255).astype(np.uint8))
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8))

    # Ensure that init_image and mask_image are in the correct format for the pipeline
    init_image = init_image_pil.convert("RGB")
    mask_image = mask_image_pil.convert("L")

    # Perform inpainting
    with torch.no_grad():
        result = pipe(image=init_image,
                    mask_image=mask_image,
                    example_image=condition_image,
                    num_inference_steps=50).images[0]

    # Save the result
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    result.save(os.path.join(output_dir, "inpainted_image.png"))

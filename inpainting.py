import os
import json
import numpy as np
import torch
from PIL import Image
import cv2
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel

# Load the Stable Diffusion Inpainting model
# base_dir = '/home/student.unimelb.edu.au/xueyangk'
# folder_type = 'fast-DiT/data/realestate/5'
# src_type = 'rgb'
# file_type = 'warped-output'
# output_folder = 'inpaint-output'

base_dir = '/home/student.unimelb.edu.au/xueyangk'
# folder_type = 'fast-DiT/data/real-estate/rgb'
# img_folder_type = 'rgb'
# filename = 'frame_000440.jpg'
# filename = '86352933.png'
# src_vae_features = 'frame_000440.npy'

# filename2 = 'frame_000470.jpg'
# filename2 = '87654233.png'
# tar_vae_features = 'frame_000470.npy'

folder_type = 'fast-DiT/data/scannet-samples/scene0560_00'
file_type = 'warped-output'
src_type = 'rgb'
depth_folder = 'depth'
pose_folder = 'c2w'
output_folder = 'warped-output'
# json_file = os.path.join(base_dir, folder_type, 'pose.json')


# Set up cache directory for models
cache_dir = os.path.join(base_dir, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)

# Load the Stable Diffusion Inpainting model
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    cache_dir=os.path.join(cache_dir, 'stable-diffusion-inpainting')
)
pipe = pipe.to("cuda")
# Load the CLIP model and processor for condition embedding
clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda")
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# Load the JSON file
# with open(json_file, 'r') as f:
#     poses_data = json.load(f)
    # print(data)
filenames = []

# Iterate over files in the directory
for filename in os.listdir(os.path.join(base_dir, folder_type, file_type)):
    if filename.endswith('.png'):
        filenames.append(filename)

# Sort filenames (assuming they are timestamps)
filenames.sort()

for filename in filenames:          
    if filename.endswith('.png'):
        frame_id = filename.split('.')[0]
    # Load the warped image to inpaint
    init_image_path = os.path.join(base_dir, folder_type, file_type, str(frame_id) + '.png')
    init_image = Image.open(init_image_path).convert("RGB")
    init_image_np = np.array(init_image)

    # Create the inverted mask (white pixels are to be inpainted)
    gray_image = cv2.cvtColor(init_image_np, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(binary_mask)
    mask_image = Image.fromarray(inverted_mask).convert("L")

    # Load the source image for condition embedding
    condition_image_path = os.path.join(base_dir, folder_type, src_type, '0.png')
    condition_image = Image.open(condition_image_path).convert("RGB")

    # Generate CLIP embedding for the condition image
    inputs = clip_processor(images=condition_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        clip_embedding = clip_model.get_image_features(**inputs)

    # Ensure clip_embedding has the correct shape
    clip_embedding = clip_embedding.unsqueeze(0)

    # Convert CLIP embedding to a text description (this is a workaround)
    prompt = "An indoor room scene with details similar to the reference condition image"

    # Perform inpainting
    with torch.no_grad():
        result = pipe(prompt=prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=50,
                    guidance_scale=7.5).images[0]

    # Save the result
    output_dir = os.path.join(base_dir, output_folder, 'scene0560_00')
    os.makedirs(output_dir, exist_ok=True)
    result.save(os.path.join(output_dir,  str(frame_id) + '.png'))

    print("Inpainting completed and image saved.")
    # import os
# import json
# import numpy as np
# import torch
# from PIL import Image
# import cv2
# from diffusers import StableDiffusionInpaintPipeline
# from transformers import CLIPProcessor, CLIPModel

# # Load the Stable Diffusion Inpainting model
# base_dir = '/home/student.unimelb.edu.au/xueyangk'
# folder_type = 'fast-DiT/data/realestate/1'
# src_type = 'rgb'
# file_type = 'warped-output'
# output_folder = 'inpaint-output'
# json_file = os.path.join(base_dir, folder_type, 'pose.json')

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# # Load the CLIP model and processor for condition embedding
# clip_model_id = "openai/clip-vit-large-patch14"
# clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda")
# clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# # Load the JSON file
# with open(json_file, 'r') as f:
#     poses_data = json.load(f)

# # Load the input image to inpaint
# init_image_path = os.path.join(base_dir, folder_type, file_type, '295728767.png')
# init_image = Image.open(init_image_path).convert("RGB")
# init_image_np = np.array(init_image)

# # Create the mask (assuming black pixels are to be inpainted)
# gray_image = cv2.cvtColor(init_image_np, cv2.COLOR_RGB2GRAY)
# _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
# inverted_mask = cv2.bitwise_not(binary_mask)
# mask_image = Image.fromarray(inverted_mask).convert("L")

# # Load the source image for condition embedding
# condition_image_path = os.path.join(base_dir, folder_type, src_type, '295395100.png')
# condition_image = Image.open(condition_image_path).convert("RGB")

# # Generate CLIP embedding for the condition image
# inputs = clip_processor(images=condition_image, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     clip_embedding = clip_model.get_image_features(**inputs)

# # Ensure clip_embedding has the correct shape
# clip_embedding = clip_embedding.unsqueeze(0)

# # Create a dummy negative prompt embedding
# negative_prompt_embeds = torch.zeros_like(clip_embedding)

# # Generate a text prompt based on the condition image
# prompt = "An indoor room scene, similar to the reference image"

# # Perform inpainting
# with torch.no_grad():
#     result = pipe(prompt=prompt,
#                   image=init_image,
#                   mask_image=mask_image,
#                   num_inference_steps=50,
#                   guidance_scale=7.5).images[0]

# # Save the result
# output_dir = os.path.join(base_dir, output_folder)
# os.makedirs(output_dir, exist_ok=True)
# result.save(os.path.join(output_dir, "inpainted_image1.png"))

# print("Inpainting completed and image saved.")

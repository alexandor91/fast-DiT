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
def center_crop_img_and_resize(src_image, image_size):
    """
    Center cropping implementation, modified to work with PIL images.
    """
    while min(src_image.size) >= 2 * image_size:
        new_size = (src_image.size[0] // 2, src_image.size[1] // 2)
        src_image = src_image.resize(new_size, Image.ANTIALIAS)

    scale = image_size / min(src_image.size)
    new_size = (round(src_image.size[0] * scale), round(src_image.size[1] * scale))
    src_image = src_image.resize(new_size, Image.BICUBIC)

    crop_y = (src_image.size[1] - image_size) // 2
    crop_x = (src_image.size[0] - image_size) // 2
    cropped_image = src_image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))

    return cropped_image

base_dir = '/home/student.unimelb.edu.au/xueyangk'
# folder_type = 'fast-DiT/data/real-estate/rgb'
# img_folder_type = 'rgb'
# filename = 'frame_000440.jpg'
# filename = '86352933.png'
# src_vae_features = 'frame_000440.npy'

# filename2 = 'frame_000470.jpg'
# filename2 = '87654233.png'
# tar_vae_features = 'frame_000470.npy'

folder_type = 'fast-DiT/data/scannet-samples/scene0181_01'
folder_type = 'fast-DiT/data/realestate/5'
file_type = 'warped-output'
src_type = 'rgb'
depth_folder = 'depth'
pose_folder = 'c2w'
output_folder = 'inpaint-output'
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
    init_image = center_crop_img_and_resize(init_image, 256)
    init_image_np = np.array(init_image)

    # Create the inverted mask (white pixels are to be inpainted)
    gray_image = cv2.cvtColor(init_image_np, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(binary_mask)
    mask_image = Image.fromarray(inverted_mask).convert("L")

    # Load the source image for condition embedding
    condition_image_path = os.path.join(base_dir, folder_type, src_type, '176276100.png')
    condition_image = Image.open(condition_image_path).convert("RGB")
    condition_image = center_crop_img_and_resize(condition_image, 256)

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
    output_dir = os.path.join(base_dir, output_folder, '2')
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

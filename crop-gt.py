import os
import cv2
import numpy as np

def center_crop_img_and_resize(src_image, image_size):
    """
    Center cropping implementation, modified to work with OpenCV images.
    """
    while min(src_image.shape[:2]) >= 2 * image_size:
        new_size = (src_image.shape[1] // 2, src_image.shape[0] // 2)
        src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_AREA)

    scale = image_size / min(src_image.shape[:2])
    new_size = (round(src_image.shape[1] * scale), round(src_image.shape[0] * scale))
    src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_CUBIC)

    crop_y = (src_image.shape[0] - image_size) // 2
    crop_x = (src_image.shape[1] - image_size) // 2
    cropped_image = src_image[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

    return cropped_image

def process_images(image_folder, output_folder, image_size=256):
    # Get list of image files (JPG or PNG)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        frame_id = image_file.split('.')[0]
        if image is not None:
            cropped_image = center_crop_img_and_resize(image, image_size)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, cropped_image)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Failed to read image: {image_path}")

if __name__ == "__main__":
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/realestate/2'
    file_type = 'rgb'
    folder_type = 'fast-DiT/data/scannet-samples/scene0181_01'
    file_type = 'rgb'
    depth_folder = 'depth'
    pose_folder = 'c2w'
    output_folder = 'cropped_images'
    image_folder = os.path.join(base_dir, folder_type, file_type)  # Change this to the correct image folder
    output_folder = os.path.join(base_dir, output_folder, 'scene0181_01')  # Change this to the desired output folder

    process_images(image_folder, output_folder)
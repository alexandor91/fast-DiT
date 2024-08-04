import cv2
import os

def images_to_video(image_folder, output_video, fps=20):
    # Get list of image files
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Ensure the images are in order

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write out frame to video

    video.release()  # Release the VideoWriter

# Example usage:

base_dir = '/home/student.unimelb.edu.au/xueyangk'
folder_type = 'fast-DiT/data/realestate/5'
file_type = 'rgb'
folder_type = 'fast-DiT/data/scannet/scene0806_00'
file_type = 'color'
depth_folder = 'depth'
pose_folder = 'c2w'
output_dir = os.path.join(base_dir, 'videos')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_folder = os.path.join(base_dir, folder_type, file_type)
output_video = 'scene0806_00.mp4'
images_to_video(image_folder, os.path.join(output_dir, output_video), fps=15)
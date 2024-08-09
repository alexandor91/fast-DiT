import json
import os
import numpy as np

# Define paths
base_dir = '/home/student.unimelb.edu.au/xueyangk'
folder_type = 'fast-DiT/data/realestate/3'
file_type = 'rgb'
output_folder = 'poses-files'
# Load the JSON file
json_file = 'pose.json'
json_path = os.path.join(base_dir, folder_type, json_file)

# Load the JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Get the list of filenames
filenames = [filename for filename in os.listdir(os.path.join(base_dir, folder_type, file_type)) if filename.endswith('.png')]
filenames.sort()

# Prepare the output file
output_dir = os.path.join(base_dir, output_folder)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'poses3.txt')

# Function to extract and write parameters
def extract_and_write_parameters(filenames, data, output_file):
    with open(output_file, 'w') as file:

        for filename in filenames:
            frame_id = filename.split('.')[0]
            for entry in data:
                timestamp = str(entry['timestamp'])
                if timestamp == frame_id:
                    intrinsics = entry['intrinsics']
                    pose = entry['pose']
                    row = [timestamp]
                    row += [intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]]
                    row += [item for sublist in pose[0:3] for item in sublist]
                    file.write(' '.join(map(str, row)) + '\n')
                    break

# Extract and write parameters to the output file
extract_and_write_parameters(filenames, data, output_file)

print("Parameters have been extracted and saved to:", output_file)
#########scannet convert code below#############
# Define paths
# base_dir = '/home/student.unimelb.edu.au/xueyangk'
# folder_type = 'fast-DiT/data/realestate/5'
# file_type = 'rgb'
# folder_type = 'fast-DiT/data/scannet-samples/scene0181_01'
# file_type = 'rgb'
# depth_folder = 'depth'
# pose_folder = 'c2w'
# output_folder = 'poses-files'
# # Load the JSON file
# json_file = 'scene_data.json'
# json_path = os.path.join(base_dir, folder_type, json_file)

# # Load the JSON file
# # with open(json_path, 'r') as file:
# #     data = json.load(file)

# # Get the list of filenames
# filenames = [filename for filename in os.listdir(os.path.join(base_dir, folder_type, file_type)) if filename.endswith('.png')]
# filenames.sort()

# # Prepare the output file
# output_dir = os.path.join(base_dir, output_folder)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# output_file = os.path.join(output_dir, 'poses-scene0181_01.txt')

# # Function to extract and write parameters
# def extract_and_write_parameters(filenames, output_file):
#     with open(output_file, 'w') as file:
#         for filename in filenames:
#             print(filename)
#             frame_id = filename.split('.')[0]

#             pose = np.load(os.path.join(base_dir, folder_type, pose_folder, str(frame_id) + '.npy'))
#             intrinsics = np.load(os.path.join(base_dir, folder_type, 'intrinsic.npy'))[:3, :3]
#             timestamp = str(frame_id)
#             row = [timestamp]
#             row += [intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]]
#             row += [item for sublist in pose[0:3] for item in sublist]
#             file.write(' '.join(map(str, row)) + '\n')

# # Extract and write parameters to the output file
# extract_and_write_parameters(filenames, output_file)

# print("Parameters have been extracted and saved to:", output_file)
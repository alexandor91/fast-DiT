import json
import os
import numpy as np

# Define paths
base_dir = '/home/student.unimelb.edu.au/xueyangk'
folder_type = 'fast-DiT/data/realestate/1'
file_type = 'rgb'
output_folder = 'warped-output'
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
output_file = os.path.join(base_dir, folder_type, 'intrinsic_extrinsic_parameters.txt')

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
                    row += [item for sublist in pose for item in sublist]
                    file.write(' '.join(map(str, row)) + '\n')
                    break

# Extract and write parameters to the output file
extract_and_write_parameters(filenames, data, output_file)

print("Parameters have been extracted and saved to:", output_file)
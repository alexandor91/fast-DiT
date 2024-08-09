import json
import numpy as np
import os


folder_type = 'fast-DiT/data/realestate/1'
file_type = 'rgb'
output_folder = 'warped-output'
base_dir = '/home/student.unimelb.edu.au/xueyangk'
folder_type = 'poses-files'
file_type = 'poses-scene0616_00.txt'

def orb_to_blender(orb_t):
    # blender start with camera looking down, -z forward
    # orb starts with camera looking forward +z forward
    pre_conversion = np.array([ # orb starts with +z forward, +y down
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1],
    ])
    conversion = np.array([ # converts +y down world to z+ up world
        [1,0,0,0],
        [0,0,1,0],
        [0,-1,0,0],
        [0,0,0,1],
    ])
    camera_local = np.linalg.inv(orb_t)
    orb_world = np.matmul(camera_local,pre_conversion)
    blender_world = np.matmul(conversion,orb_world)
    return blender_world

def convert_poses_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        values = [float(x) for x in line.strip().split()]
        focal_x = values[1]
        focal_y = values[2]
        center_x = values[3]
        center_y = values[4]
        # print(focal_x)
        orb_t = np.array([
            values[5:9],
            values[9:13],
            values[13:17],
            [0, 0, 0, 1.0]
        ])

        ###########scannet used only for inverse extrinsic pose######
        orb_t = np.linalg.inv(orb_t)
        
        blender_pose = orb_to_blender(orb_t)
        pose = blender_pose.tolist()
        poses.append(pose)

    num_frames = len(poses)
    # dependencies = list(range(num_frames - 1))
    dependencies = [None] + list(range(num_frames - 1))
    generation_order = list(range(1, num_frames))

    output_data = {
        "focal_x": focal_x,
        "focal_y": focal_y,
        "center_x": center_x,
        "center_y": center_y,
        "poses": poses,
        "dependencies": dependencies,
        "generation_order": generation_order
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

# Usage
file_type = 'poses-scene0616_00.txt'
output_file = "poses-scene0616_00.json"
file_dir = os.path.join(base_dir, folder_type, file_type)
output_dir = os.path.join(base_dir, folder_type, output_file)
convert_poses_to_json(file_dir, output_dir)
import numpy as np
# from dinov2.models.vision_transformer import vit_large  
import os
import einops as E
from PIL import Image 
from sklearn.manifold import TSNE
import umap
import cv2
import matplotlib.pyplot as plt
import json
# import zoedepth  # Assuming ZoeDepth is a hypothetical package

# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

def compute_relative_transform(src_pose, tar_pose):
    """
    Compute the relative transformation matrix from source to target view.
    
    Args:
        src_pose (np.ndarray): Source camera pose (4x4 matrix).
        tar_pose (np.ndarray): Target camera pose (4x4 matrix).
        
    Returns:
        np.ndarray: Relative transformation matrix (4x4 matrix).
    """
    relative_transform = np.dot(np.linalg.inv(tar_pose), src_pose)
    return relative_transform

def depth_to_3d_points(depth_map, intrinsic_matrix):
    """
    Convert depth map to 3D points using the camera intrinsic matrix.
    
    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        
    Returns:
        np.ndarray: 3D points of shape (H*W, 3).
    """
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    x = (j - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (i - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map

    points_3D = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points_3D

def transform_3d_points(points, relative_transform):
    """
    Apply a 3D transformation to the points.
    
    Args:
        points (np.ndarray): 3D points of shape (N, 3).
        relative_transform (np.ndarray): Relative transformation matrix (4x4).
        
    Returns:
        np.ndarray: Transformed 3D points of shape (N, 3).
    """
    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed_points_homogeneous = np.dot(points_homogeneous, relative_transform.T)
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3:]
    return transformed_points

def project_points(points, intrinsic_matrix):
    """
    Project 3D points onto the 2D image plane using the camera intrinsic matrix.
    
    Args:
        points (np.ndarray): Transformed 3D points of shape (N, 3).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        
    Returns:
        np.ndarray: 2D projected points of shape (N, 2).
    """
    points_2d_homogeneous = np.dot(points, intrinsic_matrix.T)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    return points_2d

def populate_depth_image(projected_points_2D, depth_values, H, W):
    """
    Populate the depth image with the depth values of the projected points.
    
    Args:
        projected_points_2D (np.ndarray): Projected 2D points of shape (N, 2).
        depth_values (np.ndarray): Corresponding depth values of shape (N,).
        H (int): Height of the output depth image.
        W (int): Width of the output depth image.
        
    Returns:
        np.ndarray: Populated depth image of shape (H, W).
    """
    depth_image = np.zeros((H, W), dtype=np.float32)

    x_coords = projected_points_2D[:, 0].astype(int)
    y_coords = projected_points_2D[:, 1].astype(int)

    valid_mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)

    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_depth_values = depth_values[valid_mask]

    depth_image[valid_y, valid_x] = valid_depth_values

    return depth_image

def warp_depth_map(src_depth_map, src_intrinsic, src_pose, tar_intrinsic, tar_pose):
    """
    Warp the source depth map to the target view using camera intrinsic and extrinsic parameters.
    
    Args:
        src_depth_map (np.ndarray): Source depth map.
        src_intrinsic (np.ndarray): Source camera intrinsic matrix.
        src_pose (np.ndarray): Source camera pose (4x4 matrix).
        tar_intrinsic (np.ndarray): Target camera intrinsic matrix.
        tar_pose (np.ndarray): Target camera pose (4x4 matrix).
        
    Returns:
        np.ndarray: Warped depth map in the target view.
    """
    H, W = src_depth_map.shape
    
    # Step 1: Convert depth map to 3D points
    points_3D = depth_to_3d_points(src_depth_map, src_intrinsic)
    
    # Step 2: Compute relative transform from source to target
    relative_transform = compute_relative_transform(src_pose, tar_pose)
    
    # Step 3: Transform 3D points to the target view
    transformed_points_3D = transform_3d_points(points_3D, relative_transform)
    
    # Step 4: Project transformed 3D points to 2D in the target view
    projected_points_2D = project_points(transformed_points_3D, tar_intrinsic)
    
    # Step 5: Populate the depth image in the target view
    warped_depth_map = populate_depth_image(projected_points_2D, transformed_points_3D[:, 2], H, W)
    
    return warped_depth_map

def load_resize_image_cv2(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    image = image.astype(np.float32)
    return image

if __name__ == "__main__":
    # Download all DiT checkpoints
    print("###### main starts#########")

    base_dir = '/home/student.unimelb.edu.au/xueyangk'

    folder_type = 'fast-DiT/data/realestate/1'
    file_type = 'rgb'
    output_folder = 'warped-depth'
    # Load the JSON file
    json_file = 'pose.json'
    with open(os.path.join(base_dir, folder_type, json_file), 'r') as file:
        # file = os.path.join(base_dir, folder_type, json_file)
        data = json.load(file)
    # print(data)
    filenames = []

    # Iterate over files in the directory
    for filename in os.listdir(os.path.join(base_dir, folder_type, file_type)):
        if filename.endswith('.png'):
            filenames.append(filename)
    
    # Sort filenames (assuming they are timestamps)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    print(filenames_sorted)

    src_intrinsic = None
    tar_intrinsic = None
    src_homo_mat_sample = None
    tar_homo_mat_sample = None

    # src_feats_path = os.path.join(base_dir, folder_type, f'{frame_id}.npy')
    # src_feats = np.load(src_feats_path)
    # ###########load VAE numpy features################
    # src_feats = np.load(os.path.join(base_dir, folder_type, src_vae_features))
    # tar_feats = np.load(os.path.join(base_dir, folder_type, tar_vae_features))

    src_image_path = os.path.join(base_dir, folder_type, file_type, filenames[0])
    src_image = load_resize_image_cv2(src_image_path)    
    # src_image = center_crop_img_and_resize(src_image)
    src_frame_id = str(filenames[0].split('.')[0])
    print(src_frame_id)

    # Iterate through the data to find the specific timestamp
    for entry in data:
        # print("#####found $$$$$$$##########")
        timestamp = int(entry['timestamp'])
        print("$$$$$$  json file  $$$$$$$")
        print(timestamp)
        #     pose = entry['pose']
        if timestamp == int(src_frame_id):
            print("#####found $$$$$$$##########")
            src_homo_mat_sample = np.array(entry['pose'])
            src_intrinsic = np.array(entry['intrinsics'])
            src_intrinsic[0, :] = src_intrinsic[0, :] * src_image.shape[1] 
            src_intrinsic[1, :] = src_intrinsic[1, :] * src_image.shape[0] 
            scale1 = 256/min(src_image.shape[:2]) #########final feature map size is 32
            scale2 = 256/(min(src_image.shape[:2]) * (src_intrinsic[1, 1]/src_intrinsic[0, 0]))
            # src_intrinsic[0, 0] = src_intrinsic[0, 0] * scale1
            # src_intrinsic[1, 1] = src_intrinsic[1, 1] * scale2
            # src_intrinsic[0, 2] = 128  #954.7021
            # src_intrinsic[1, 2] = 128  #723.6698
            #########downsampling to 32 x 32 feature map
            break
        else: 
            continue
    
    print("#####1 $$$$$$$##########")
    print(src_homo_mat_sample)
    filenames_sorted.pop(0)  # Remove the first file from the list
    for filename in filenames_sorted:          
        if filename.endswith('.png'):
            frame_id = filename.split('.')[0]
        if int(frame_id) == int(src_frame_id):
            continue
        for entry in data:
            timestamp = int(entry['timestamp'])
            if timestamp == int(frame_id):
                tar_homo_mat_sample = np.array(entry['pose'])
                tar_intrinsic = np.array(entry['intrinsics'])
                tar_intrinsic[0, :] = tar_intrinsic[0, :] * src_image.shape[1] 
                tar_intrinsic[1, :] = tar_intrinsic[1, :] * src_image.shape[0] 
                scale1 = 256/min(src_image.shape[:2]) #########final feature map size is 32
                scale2 = 256/(min(src_image.shape[:2]) * (tar_intrinsic[1, 1]/tar_intrinsic[0, 0]))
                # tar_intrinsic[0, 0] = tar_intrinsic[0, 0] * scale1
                # tar_intrinsic[1, 1] = tar_intrinsic[1, 1] * scale2
                # tar_intrinsic[0, 2] = 128    #954.7021
                # tar_intrinsic[1, 2] = 128    #723.6698
                break
        
        print("#####1 $$$$$$$##########")
        print(tar_homo_mat_sample)
        # tgt_image_path = os.path.join(base_dir, folder_type, file_type, filename)
        # tgt_image = load_resize_image_cv2(tgt_image_path)
        # tgt_image = center_crop_img_and_resize(tgt_image)

        # Define the target timestamp
        # src_timestamp = 86352933 # Example timestamp, replace with the desired one
        # tar_timestamp = 87654233
        # Initialize variables for pose and intrinsics

        # depth_file = '86352933_depth.npy'
        # folder_type = 'fast-DiT/data/real-estate/depth'
        # depth_map = cv2.imread(os.path.join(base_dir, folder_type, depth_file), cv2.IMREAD_UNCHANGED)   
        file_type = 'depth'
        print("########$$$$$$$$$$$$$##########")
        print(os.path.join(base_dir, folder_type, file_type, str(frame_id) + '.npy'))
        # Load corresponding depth map
        frame_id = filename.split('.')[0]
        depth_map_path = os.path.join(base_dir, folder_type, file_type, frame_id + '.npy')
        depth_map = np.load(depth_map_path)[0][0]
        # depth_map = center_crop_img_and_resize(depth_map, 256)

        print(depth_map.shape)
        print(src_image.shape)
        if depth_map is None:
            raise ValueError(f"Failed to load depth map")

        # if depth_map.dtype != np.uint16 or depth_map.dtype != np.uint8:
        #     raise ValueError("Depth map is not in uint16 format")
        depth_map_float = depth_map.astype(np.float32)


        # Check if matrices were found
        if src_homo_mat_sample is not None and src_intrinsic is not None:
            print("Pose Matrix:")
            print(src_homo_mat_sample)
            print("\nIntrinsics Matrix:")
            print(src_intrinsic)

        # src_homo_mat_sample = np.linalg.inv(src_homo_mat_sample)
        # tar_homo_mat_sample = np.linalg.inv(tar_homo_mat_sample)
        relative_homo_mat = np.dot(tar_homo_mat_sample, np.linalg.inv(src_homo_mat_sample))

        # _, C, H, W = src_feats.shape
        H, W, C = src_image.shape
        factor = H / depth_map_float.shape[0]
        # factor2 = W / depth_map_float.shape[1]
        # assert factor1 == factor2, f"Factors are not equal: factor1 = {factor1}, factor2 = {factor2}"
        # factor = factor2
        # new_H = int(H / factor)
        # new_W = int(W / factor)

        # downsampled_image = cv2.resize(np.transpose(src_image, (1, 2, 0)), (new_W, new_H))
        # depth_map_float = cv2.resize(depth_map_float, (W, H), interpolation=cv2.INTER_CUBIC)  ### cv2.INTER_LINEAR
        # src_feats = src_feats[0]
        # tar_feats = tar_feats[0]    

        warped_depth_map = warp_depth_map(depth_map, src_intrinsic, src_homo_mat_sample, tar_intrinsic, tar_homo_mat_sample)

        # Check if the directory exists, create if it doesn't
        output_dir = os.path.join(base_dir, folder_type, output_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # # Define the full path for the file
        save_path = os.path.join(output_dir, str(frame_id) + '.png')
                
        # # # Example 2D NumPy array (depth map)
        # depth_map = np.random.rand(H, W)  # Replace with your actual data

        # # Normalize the depth map to the range [0, 1]
        # depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # # Apply the magma colormap
        # colored_image = plt.cm.magma(depth_normalized)

        # # Save the colored image
        # plt.imsave(save_path, warped_depth_map)
        # print("#######depth result saved#######")
        print(warped_depth_map)
        cv2.imwrite(save_path, warped_depth_map)

        # project_and_save_tsne_image(warped_image, 'tsne_warped_viz.png')

    # fig, ax = plt.subplots()

    # # Display depth map with jet colormap
    # im = ax.imshow(depth_map, cmap='magma')
    ## plt.colorbar(im, ax=ax)  # Add colorbar for reference

    # Save the displayed image with jet colormap
    # plt.savefig('depth_map_magma_color.png')
    # plt.close(fig)


    # depth_map_float /= 255.0
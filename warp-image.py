import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
# from dinov2.models.vision_transformer import vit_large 
from PIL import Image 
import torchvision.transforms as transforms 
import os
import einops as E
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import umap
from torch import nn, einsum
from einops import rearrange, repeat
import cv2
import matplotlib.pyplot as plt


def quaternion_to_rotation_matrix(quaternions): #############tensor version#############
    """
    Converts a batch of quaternions to rotation matrices.
    
    Args:
        quaternion (torch.Tensor): Tensor of shape (batch_size, 4)
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3, 3)
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    batch_size = quaternions.size(0)
    return torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=1),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=1),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2], dim=1)
    ], dim=1).reshape(batch_size, 3, 3)

def compute_relative_transform(src_pose, tar_pose):
    # Compute the relative transformation matrix
    relative_transform = torch.bmm(tar_pose, torch.inverse(src_pose))
    return relative_transform

def warp_image(source_image, intrinsic_matrix, src_pose, tar_pose):
    # Get the batch size
    batch_size = source_image.shape[0]

    # Compute the relative transformation
    relative_transform = compute_relative_transform(src_pose, tar_pose)  # shape [B, 4, 4]

    # Decompose the intrinsic matrix
    K = intrinsic_matrix  # shape [B, 3, 3]
    K_inv = torch.inverse(K)  # shape [B, 3, 3]

    # Extract rotation (R) and translation (t)
    R = relative_transform[:, :3, :3]  # shape [B, 3, 3]
    t = relative_transform[:, :3, 3:]  # shape [B, 3, 1]

    # Compute homography H = K * (R - t * n^T / d) * K_inv
    # For simplicity, we assume plane at z=1 (n = [0, 0, 1], d = 1)
    n_T = torch.tensor([[0, 0, 1]], device=source_image.device, dtype=torch.float32).repeat(batch_size, 1, 1)  # shape [B, 1, 3]
    t_skew = t @ n_T  # shape [B, 3, 3]
    t_skew_adjusted = R - t_skew  # shape [B, 3, 3]

    # Compute the homography
    H = torch.bmm(K, torch.bmm(t_skew_adjusted, K_inv))  # shape [B, 3, 3]

    # Create a grid of coordinates (u, v) in the target image
    height, width = source_image.shape[2], source_image.shape[3]
    u, v = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32))
    u, v = u.to(source_image.device), v.to(source_image.device)
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=2)  # shape [H, W, 3]

    # Flatten the grid to apply the transformation
    uv1_flat = uv1.reshape(-1, 3).T.unsqueeze(0).repeat(batch_size, 1, 1).float()  # shape [B, 3, H*W]

    # Ensure homography matrix is of type float32
    H = H.float()

    # Project the grid coordinates using the homography transformation
    projected = torch.bmm(H, uv1_flat)  # shape [B, 3, H*W]

    # Normalize the projected coordinates
    projected = projected / projected[:, 2:3, :]  # shape [B, 3, H*W]
    projected_uv = projected[:, :2, :].reshape(batch_size, 2, height, width)  # shape [B, 2, H, W]

    # Normalize the projected coordinates to [-1, 1] range for grid_sample
    projected_uv[:, 0, :, :] = (projected_uv[:, 0, :, :] / (width - 1)) * 2 - 1  # Normalize x to [-1, 1]
    projected_uv[:, 1, :, :] = (projected_uv[:, 1, :, :] / (height - 1)) * 2 - 1  # Normalize y to [-1, 1]
    projected_uv = projected_uv.permute(0, 2, 3, 1)  # shape [B, H, W, 2]

    print("########projected pixels!!########")
    print(projected_uv)

    # Initialize the output tensor for the warped image
    warped_image = torch.zeros_like(source_image)

    # Warp each channel independently
    for c in range(source_image.shape[1]):
        # Warp the current channel of the source image using the projected coordinates
        warped_channel = F.grid_sample(source_image[:, c:c+1, :, :], projected_uv, mode='bilinear', padding_mode='zeros', align_corners=True)  # shape [B, 1, H, W]
        warped_image[:, c:c+1, :, :] = warped_channel

    return warped_image


def clean_feature_map(feature_map_flat):
    # Remove rows with NaNs or Infs
    valid_indices = ~np.any(np.isnan(feature_map_flat) | np.isinf(feature_map_flat), axis=1)
    cleaned_feature_map = feature_map_flat[valid_indices]
    return cleaned_feature_map, valid_indices

def visualize_feature_map(feature_map, output_path):
    # Select the feature map of interest (batch index 0)
    feature_map = feature_map[0]  # shape [4, 32, 32]

    # Reshape the feature map to [H*W, C]
    height, width = feature_map.shape[1], feature_map.shape[2]
    feature_map_flat = feature_map.permute(1, 2, 0).reshape(-1, 4).cpu().numpy()  # shape [H*W, C]

    # Apply t-SNE to reduce to 3 dimensions
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    try:
        feature_map_3d = tsne.fit_transform(feature_map_flat)  # shape [HxW, 3]
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        return
    # Clean the feature map to remove NaNs and Infs
    feature_map_cleaned, valid_indices = clean_feature_map(feature_map_3d)

    # Check if all data points were invalid
    if feature_map_cleaned.size == 0:
        raise ValueError("All data points are invalid (NaNs or Infs).")

    # Create an empty image and fill in the valid points
    feature_map_3d_image = np.zeros((height * width, 3))
    feature_map_3d_image[valid_indices] = feature_map_3d

    # Reshape back to image shape
    feature_map_3d_image = feature_map_3d_image.reshape(height, width, 3)

    # Normalize the feature map to the range [0, 255]
    min_val = feature_map_3d_image.min()
    max_val = feature_map_3d_image.max()
    if max_val > min_val:
        feature_map_3d_image = (feature_map_3d_image - min_val) / (max_val - min_val)
    feature_map_3d_image = (feature_map_3d_image * 255).astype(np.uint8)

    # Save the image as a PNG file
    cv2.imwrite(output_path, feature_map_3d_image)
    print(f"Feature map image saved to {output_path}")


if __name__ == "__main__":
    # Download all DiT checkpoints
    print("###### main starts#########")
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data'
    img_folder_type = 'rgb'
    filename = 'frame_000440.jpg'
    src_vae_features = 'frame_000440.npy'

    filename2 = 'frame_000470.jpg'
    tar_vae_features = 'frame_000470.npy'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 6
    src_feats = torch.from_numpy(np.load(os.path.join(base_dir, folder_type, src_vae_features))).to(device)
    src_feats = src_feats.repeat(batch_size, 1, 1, 1)
    tar_feats = torch.from_numpy(np.load(os.path.join(base_dir, folder_type, tar_vae_features))).to(device)
    tar_feats = tar_feats.repeat(batch_size, 1, 1, 1)


    ###############pre processing for dino features ############################
    # image = Image.open(os.path.join(base_dir, folder_type, filename)) 
    
    # # Define a transform to convert PIL  image to a Torch tensor 
    # transform = transforms.Compose([ 
    #     transforms.PILToTensor() 
    # ])

    # # Define the transformations
    # preprocess = transforms.Compose([
    #     transforms.Resize(512, interpolation=Image.BILINEAR),
    #     transforms.CenterCrop(512),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # # Define the batch size
    # b = 8
    # feature_dim = 4 
    # img_height = 32 
    # img_width = 32
    # # Initialize random tensors
    # K = torch.randn(b, 3, 3, device='cuda', dtype=torch.float32)
    # R = torch.randn(b, 3, 3, device='cuda', dtype=torch.float32)
    # t = torch.randn(b, 3, 1, device='cuda', dtype=torch.float32)
    # f_src_flat = torch.randn(b, 32, 32, 4, device='cuda', dtype=torch.float32)
    # tar_proj = torch.randn(b, 32, 32, 4, device='cuda', dtype=torch.float32)
    # N = img_width*img_height  # Example value for N, you can change it as needed

    # # Initialize random tensors
    # f_src_flat = torch.randn(b, 3, N, device='cuda')
    # tar_proj = torch.randn(b, 3, N, device='cuda')

    # print("########  main #########")
    # print(f_src_flat.shape)
    # print(tar_proj.shape)
    # o_proj = torch.randn(b, 3, N, device='cuda')
    # base_dir = '/home/student.unimelb.edu.au/xueyangk/fast-DiT'
    # source_img = cv2.imread(os.path.join(base_dir, folder_type, 'frame_000440.jpg'))
    # target_img = cv2.imread(os.path.join(base_dir, folder_type, 'frame_000470.jpg'))

    # Define the camera parameters and transformation matrices
    scene_id = '0a5c013435'
    focal_length_x = 1432.3682
    focal_length_y = 1432.3682
    principal_point_x = 954.08276
    principal_point_y = 724.18256

    # Intrinsic matrices
    src_intrinsic = np.array([[focal_length_x, 0, principal_point_x],
                            [0, focal_length_y, principal_point_y],
                            [0, 0, 1]])

    focal_length_x2 = 1431.4313
    focal_length_y2 = 1431.4313
    principal_point_x2 = 954.7021
    principal_point_y2 = 723.6698

    tar_intrinsic = np.array([[focal_length_x2, 0, principal_point_x2],
                            [0, focal_length_y2, principal_point_y2],
                            [0, 0, 1]])

    # Quaternions and translations
    src_quatern = np.array([0.837752, 0.490157, -0.150019, 0.188181])
    src_trans = np.array([0.158608, 1.22818, -1.60956])
    tar_quatern = np.array([0.804066, 0.472639, -0.25357, 0.256501])
    tar_trans = np.array([0.473911, 1.28311, -1.5215])

    # Homogeneous matrices
    # src_homo_mat_sample = np.array([[0.42891303, 0.40586197, 0.8070378, 1.4285464], #########aligne pose
    #                             [-0.06427293, -0.8774122, 0.4754123, 1.6330968],
    #                             [0.9010566, -0.25578123, -0.35024756, -1.2047926],
    #                             [0.0, 0.0, 0.0, 0.99999964]])
    src_homo_mat_sample = np.array([[0.42891303, 0.40586197, 0.8070378, 1.4285464], ###########raw source pose
            [-0.06427293, -0.8774122, 0.4754123, 1.6330968],
            [0.9010566, -0.25578123, -0.35024756, -1.2047926],
            [0.0, 0.0, 0.0, 0.99999964]])
    # tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ############aligned pose
    #                             [-0.1672538, -0.8868709, 0.43068102, 1.642482],
    #                             [0.966491, -0.23377006, -0.106051974, -1.1564484],
    #                             [0.0, 0.0, 0.0, 0.9999995]])
    tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ########raw target pose
            [-0.1672538, -0.8868709, 0.43068102, 1.642482],
            [0.966491, -0.23377006, -0.106051974, -1.1564484],
            [0.0, 0.0, 0.0, 0.9999995]])

    src_homo_mat = torch.tensor([src_homo_mat_sample] * batch_size, dtype=torch.float32, device=device)
    tar_homo_mat = torch.tensor([tar_homo_mat_sample] * batch_size, dtype=torch.float32, device=device)

    src_intrinsic = torch.tensor([src_intrinsic] * batch_size, dtype=torch.float32, device=device)
    tar_intrinsic = torch.tensor([tar_intrinsic] * batch_size, dtype=torch.float32, device=device)
    src_trans = torch.tensor([src_trans] * batch_size, dtype=torch.float32, device=device).view(-1, 3, 1)
    tar_trans = torch.tensor([tar_trans] * batch_size, dtype=torch.float32, device=device).view(-1, 3, 1)
    src_quaterns = torch.tensor([src_quatern] * batch_size, dtype=torch.float32, device=device)
    tar_quaterns = torch.tensor([tar_quatern] * batch_size, dtype=torch.float32, device=device)
    print("###########!!!!!!!!!!!!!!!!!!!###############")
    print(src_trans.shape)
    print(tar_trans.shape)
    print(src_homo_mat.shape)
    # Compute rotation matrices from quaternions
    # src_rot_mat = torch.linalg.inv(quaternion_to_rotation_matrix(src_quaterns)).to(device)
    # tar_rot_mat = torch.linalg.inv(quaternion_to_rotation_matrix(tar_quaterns)).to(device)
    src_rot_mat = quaternion_to_rotation_matrix(src_quaterns).to(device)
    tar_rot_mat = quaternion_to_rotation_matrix(tar_quaterns).to(device)
    # Compute relative rotation matrix
    relative_rot_mat = torch.bmm(tar_rot_mat, src_rot_mat.transpose(1, 2))
    # Compute relative homogeneous matrix
    relative_homo_mat = torch.bmm(tar_homo_mat, torch.linalg.inv(src_homo_mat))
    # relative_homo_mat = torch.matmul(tar_homo_mat, torch.inverse(src_homo_mat))

    # Example source pixel position
    u = 600
    v = 500
    src_pt = torch.tensor([u, v], dtype=torch.float32, device=device)
    print("#######relative homo######")
    # print(relative_homo_mat[:3, 3])
    print(src_feats.shape)
    print(tar_feats.shape)
    # # For tar_feats and src_feats
    # tar_feats = tar_feats.unsqueeze(0).repeat(8, 1, 1, 1)
    # src_feats = src_feats.unsqueeze(0).repeat(8, 1, 1, 1)

    # For src_intrinsic and relative_homo_mat[:3, :3]
    # src_intrinsic = src_intrinsic.unsqueeze(0).repeat(1, 1, 1)
    # relative_homo_mat_R = relative_homo_mat[:3, :3].unsqueeze(0).repeat(1, 1, 1)

    # # For relative_homo_mat[:3, 3]
    # relative_homo_mat_T = relative_homo_mat[:3, 3].unsqueeze(0).repeat(1, 1, 1)
    # relative_homo_mat_R = relative_homo_mat[:, :3, :3]

    # # For relative_homo_mat[:3, 3]
    # relative_homo_mat_T = relative_homo_mat[:, :3, 3]
    # Example usage
    # batch_size = 2
    # height, width = 128, 128
    # channels = 3

    # # Generate random data for demonstration purposes
    # source_image = torch.randint(0, 255, (batch_size, channels, height, width), dtype=torch.float32, device='cuda')
    # intrinsic_matrix = torch.eye(3, dtype=torch.float32, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)
    # src_pose = torch.eye(4, dtype=torch.float32, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)
    # tar_pose = torch.eye(4, dtype=torch.float32, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)

    # # Modify the target pose for demonstration
    # tar_pose[:, 0, 3] = 0.5  # Translate along x-axis

    # Warp the source image to the target view
    warped_image = warp_image(src_feats, src_intrinsic, src_homo_mat, tar_homo_mat)
    # print(src_feats)
    print("#######warp done########")
    batch_size = warped_image.size(0)
    
    for batch_idx in range(batch_size):
        nonzero_indices = torch.nonzero(warped_image[batch_idx], as_tuple=False)
        nonzero_values = warped_image[batch_idx][warped_image[batch_idx] != 0]
        
        print(f"Batch {batch_idx}:")
        for idx, value in zip(nonzero_indices, nonzero_values):
            print(f"Index: {idx.tolist()}, Value: {value.item()}")
    # print(warped_image)
    visualize_feature_map(warped_image, 'warped_sample_image.png')
    # # Save and show the warped images
    # for i in range(batch_size):
    #     if i == 0:
    #         save_and_show_image(warped_image[i], f'warped_image_{i}.png')

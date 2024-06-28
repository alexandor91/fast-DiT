import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
# import zoedepth  # Assuming ZoeDepth is a hypothetical package

# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

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

def depth_to_3d_points_with_colors(depth, K, src_image):
    B, H, W = depth.shape
    i, j = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
    i = i.reshape(1, H, W).expand(B, -1, -1)
    j = j.reshape(1, H, W).expand(B, -1, -1)
    
    Z = depth
    X = (j - K[:, 0, 2].view(B, 1, 1)) * Z / K[:, 0, 0].view(B, 1, 1)
    Y = (i - K[:, 1, 2].view(B, 1, 1)) * Z / K[:, 1, 1].view(B, 1, 1)
    
    points_3D = torch.stack((X, Y, Z), dim=3)  # Shape: (B, H, W, 3)
    
    # Add RGB information
    points_with_colors = torch.cat((points_3D, src_image.permute(0, 2, 3, 1)), dim=3)  # Shape: (B, H, W, 6)
    return points_with_colors

def transform_points_with_colors(points_with_colors, R, t):
    B, H, W, _ = points_with_colors.shape
    points = points_with_colors[..., :3].reshape(B, -1, 3).permute(0, 2, 1)  # Shape: (B, 3, H*W)
    colors = points_with_colors[..., 3:].reshape(B, -1, 3)  # Shape: (B, H*W, 3)
    
    points_transformed = torch.bmm(R, points) + t.unsqueeze(-1)
    points_transformed = points_transformed.permute(0, 2, 1).reshape(B, H, W, 3)  # Shape: (B, H, W, 3)
    
    # Combine transformed points with colors
    transformed_with_colors = torch.cat((points_transformed, colors.reshape(B, H, W, 3)), dim=3)  # Shape: (B, H, W, 6)
    return transformed_with_colors

def project_points_with_colors(points_with_colors, K):
    B, H, W, _ = points_with_colors.shape  # Shape: (B, H, W, 6)
    points = points_with_colors[..., :3].reshape(B, -1, 3)  # Shape: (B, H*W, 3)
    colors = points_with_colors[..., 3:].reshape(B, H, W, 3)  # Shape: (B, H, W, 3)
    
    points_2D = torch.bmm(points, K.transpose(1, 2))  # Shape: (B, H*W, 3)
    points_2D = points_2D[..., :2] / points_2D[..., 2:3]  # Shape: (B, H*W, 2)
    points_2D = points_2D.reshape(B, H, W, 2)  # Shape: (B, H, W, 2)
    
    return points_2D, colors

def warp_image(src_image, tgt_points):
    B, C, H, W = src_image.shape
    tgt_points = tgt_points.reshape(B, H, W, 2)  # Shape: (B, H, W, 2)
    
    tgt_points[..., 0] = (2.0 * tgt_points[..., 0] / (W - 1)) - 1.0
    tgt_points[..., 1] = (2.0 * tgt_points[..., 1] / (H - 1)) - 1.0
    
    # tgt_points = tgt_points.permute(0, 3, 1, 2)  # Shape: (B, 2, H, W)
    
    warped_image = F.grid_sample(src_image, tgt_points, align_corners=True)
    return warped_image

def populate_image_with_colors(projected_points_2D, colors):
    batch_size, image_height, image_width, _ = projected_points_2D.shape
    # image_height, image_width = 256, 256

    # Initialize an empty image tensor
    images = torch.zeros(batch_size, image_height, image_width, 3).to(device)

    # Get the x and y coordinates
    x_coords = projected_points_2D[..., 0]
    y_coords = projected_points_2D[..., 1]

    # Create a mask for valid coordinates
    valid_mask = (x_coords >= 0) & (x_coords < image_width) & (y_coords >= 0) & (y_coords < image_height)

    # Convert the coordinates to integer type
    x_coords = x_coords.long()
    y_coords = y_coords.long()

    # Use the valid mask to filter out-of-bound coordinates
    for b in range(batch_size):
        valid_x = x_coords[b][valid_mask[b]]
        valid_y = y_coords[b][valid_mask[b]]
        valid_colors = colors[b][valid_mask[b]]
        images[b, valid_y, valid_x] = valid_colors
    print("@@@@@@@return images@@@@@@@")
    print(images.shape)
    return images


def load_resize_image_cv2(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a tensor
    image = torch.tensor(image, dtype=torch.float32).to(device)
    # Rearrange the dimensions to (C, H, W)
    image = image.permute(2, 0, 1)
    return image

# Function to save depth map
def save_depth_map(depth_map, save_path):
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imsave(save_path, depth_map, cmap='inferno')

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

    # Define transformation for input images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # repo = 'isl-org/ZoeDepth'
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

    # Load source and target images using OpenCV
    src_image_path = os.path.join(base_dir, folder_type, filename)
    tgt_image_path = os.path.join(base_dir, folder_type, filename2)
    src_image = load_resize_image_cv2(src_image_path).repeat(batch_size, 1, 1, 1).to(device)  # Add batch dimension
    tgt_image = load_resize_image_cv2(tgt_image_path).repeat(batch_size, 1, 1, 1).to(device)  # Add batch dimension

    # # Predict depth maps
    # with torch.no_grad():
    #     depth_src = model_zoe_n(src_image)
    #     depth_tgt = model_zoe_n(tgt_image)

    # # Save depth maps
    # save_depth_map(depth_src, 'depth_src.png')
    # save_depth_map(depth_tgt, 'depth_tgt.png')
    # Example usage
    # B = 4  # Batch size
    # H, W = 256, 256  # Image dimensions

    # # Load example data
    # src_image = torch.rand((B, 3, H, W)).to(torch.float32)  # Source image
    # depth_src = torch.rand((B, H, W)).to(torch.float32)  # Source depth map
    # depth_tgt = torch.rand((B, H, W)).to(torch.float32)  # Target depth map

    # # Camera intrinsic parameters for each image in the batch
    # K = torch.tensor([[[500, 0, 128], [0, 500, 128], [0, 0, 1]]] * B).to(torch.float32)  # Example values

    # # Define the camera parameters and transformation matrices
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
    src_homo_mat_sample = np.linalg.inv(src_homo_mat_sample)

    # tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ############aligned pose
    #                             [-0.1672538, -0.8868709, 0.43068102, 1.642482],
    #                             [0.966491, -0.23377006, -0.106051974, -1.1564484],
    #                             [0.0, 0.0, 0.0, 0.9999995]])
    tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ########raw target pose
            [-0.1672538, -0.8868709, 0.43068102, 1.642482],
            [0.966491, -0.23377006, -0.106051974, -1.1564484],
            [0.0, 0.0, 0.0, 0.9999995]])
    tar_homo_mat_sample = np.linalg.inv(tar_homo_mat_sample)

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

    # Load the depth map image using OpenCV
    depth_file = 'depth_frame_000440.png'  # Replace with your depth map image path
    depth_map = cv2.imread(os.path.join(base_dir, folder_type, depth_file), cv2.IMREAD_UNCHANGED)

    # Check if the depth map was loaded properly
    if depth_map is None:
        raise ValueError(f"Failed to load depth map from {depth_file}")

    # Ensure the depth map is in uint16 format
    if depth_map.dtype != np.uint16:
        raise ValueError("Depth map is not in uint16 format")

    # Convert depth map to float32 using numpy
    depth_map_float = depth_map.astype(np.float32)

    # Normalize from millimeters to meters
    depth_map_float /= 1000.0

    # Convert to PyTorch tensor
    depth_map_tensor = torch.from_numpy(depth_map_float).to(device)
    depth_map_tensor = depth_map_tensor.repeat(batch_size, 1, 1)
    _, channels, H, W = src_image.shape
    factor1 = H / depth_map_tensor.shape[1]
    factor2 = W / depth_map_tensor.shape[2]
    assert factor1 == factor2, f"Factors are not equal: factor1 = {factor1}, factor2 = {factor2}"
    factor = factor2
    # Calculate the new height and width for downsampling
    new_H = int(H / factor)
    new_W = int(W / factor)

    # Downsample the image
    # print(src_image[0].cpu().numpy())
    downsampled_image = F.interpolate(src_image, size=(new_H, new_W), mode='bilinear', align_corners=False)
    downsampled_image_cv = downsampled_image.permute(0, 2, 3, 1)
    print(downsampled_image_cv.shape)
    cv2.imwrite('scaled-img.png', downsampled_image_cv[0].to(torch.uint8).cpu().numpy())

    print("########depth tensor#########")
    print(factor)
    print(downsampled_image.shape)
    print(depth_map_tensor.shape)
    # # Invert the intrinsic matrices
    # K_inv = torch.inverse(K
    # Convert depth map to 3D points with RGB colors in source camera frame
    points_with_colors = depth_to_3d_points_with_colors(depth_map_tensor, src_intrinsic/factor, downsampled_image)
    print(points_with_colors.shape)
    
    # Transform points with colors to target camera frame
    transformed_with_colors = transform_points_with_colors(points_with_colors, relative_homo_mat[:, :3, :3], relative_homo_mat[:, :3, 3])
    
    # Project transformed 3D points with colors onto the target camera image plane
    projected_points_2D, colors = project_points_with_colors(transformed_with_colors, tar_intrinsic/factor)
    print(projected_points_2D.shape)
    print(colors)
    # print((projected_points_2D >= 0).all(dim=-1).all())
    warped_images = populate_image_with_colors(projected_points_2D, colors)    
    # Warp the source image to the target view
    # warped_image = warp_image(downsampled_image, projected_points_tgt)

    # Display the warped image (convert to numpy for visualization)
    warped_image = warped_images[0].cpu().numpy()
    print(warped_image.shape)
    # print(warped_image)
    cv2.imwrite('Warped-img.png', warped_image)
    # # Example source pixel position
    # u = 600
    # v = 500
    # src_pt = torch.tensor([u, v], dtype=torch.float32, device=device)
    # print("#######relative homo######")
    # # print(relative_homo_mat[:3, 3])
    # print(src_feats.shape)
    # print(tar_feats.shape)

    # # Warp the source image to the target view
    # # warped_image = warp_image(src_feats, src_intrinsic, src_homo_mat, tar_homo_mat)
    # # print(src_feats)
    # print("#######warp done########")
    # batch_size = warped_image.size(0)
    
    # for batch_idx in range(batch_size):
    #     nonzero_indices = torch.nonzero(warped_image[batch_idx], as_tuple=False)
    #     nonzero_values = warped_image[batch_idx][warped_image[batch_idx] != 0]
        
    #     print(f"Batch {batch_idx}:")
    #     for idx, value in zip(nonzero_indices, nonzero_values):
    #         print(f"Index: {idx.tolist()}, Value: {value.item()}")
    # # print(warped_image)
    # visualize_feature_map(warped_image, 'warped_sample_image.png')
    # # Save and show the warped images
    # for i in range(batch_size):
    #     if i == 0:
    #         save_and_show_image(warped_image[i], f'warped_image_{i}.png')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import cv2
import os

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

def load_resize_image_cv2(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    image = cv2.resize(image, (int(width/20), int(height/20)), interpolation=cv2.INTER_CUBIC) 
    # Convert the image from BGR to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a tensor
    image = torch.tensor(image, dtype=torch.float32).to(device)
    # Rearrange the dimensions to (C, H, W)
    image = image.permute(2, 0, 1)
    return image

def compute_plucker_coordinates(extrinsic_matrix, intrinsic_matrix, H, W):
    """
    Compute Plücker coordinates for rays based on camera parameters.
    
    :param extrinsic_matrix: [4, 4] tensor representing the transform from the source frame to target frame( source_frame_pose / target_frame_pose)
    :param intrinsic_matrix: [3, 3] tensor representing the camera intrinsic matrix
    :param H: int, height of the image
    :param W: int, width of the image
    :return: Tensor of Plücker coordinates [num_rays, 6]
    """
    device = extrinsic_matrix.device
    i, j = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    i = i.float()
    j = j.float()
    
    # Intrinsic matrix components
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Compute pixel coordinates normalized by focal lengths
    u = torch.stack([(j - cx) / fx, (i - cy) / fy, torch.ones_like(j)], dim=-1).reshape(-1, 3)
    
    # Compute directions d in camera space
    K_inv = torch.inverse(intrinsic_matrix)
    directions_camera = (K_inv @ u.T).T
    
    # Normalize directions
    directions_camera = directions_camera / torch.norm(directions_camera, dim=-1, keepdim=True)

    # Extract rotation and translation from extrinsic matrix
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]
    
    # Compute directions d in world space
    directions_world = (R.T @ directions_camera.T).T

    # Compute the camera center in world coordinates
    camera_center_world = -R.T @ t

    # Compute the origins (camera centers) expanded to match number of rays
    origins_world = camera_center_world.expand(directions_world.shape)

    # Compute the moment vector m
    moments = torch.cross(origins_world, directions_world, dim=-1)

    # Combine directions and moments into Plücker coordinates
    plucker_coords = torch.cat([directions_world, moments], dim=-1)

    return plucker_coords

def visualize_plucker_coordinates_tsne(plucker_coordinates, H, W):
    """
    Visualize Plücker coordinates using t-SNE and save as an RGB image.
    
    :param plucker_coordinates: Tensor of Plücker coordinates [num_rays, 6]
    :param H: int, height of the image
    :param W: int, width of the image
    """
    # Convert to numpy array and apply t-SNE
    plucker_np = plucker_coordinates.cpu().numpy()
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(plucker_np)
    
    # Normalize the t-SNE result to fit into RGB color space
    tsne_min, tsne_max = tsne_result.min(0), tsne_result.max(0)
    tsne_norm = (tsne_result - tsne_min) / (tsne_max - tsne_min)
    tsne_norm = (tsne_norm * 255).astype(np.uint8)
    
    # Reshape to form an image
    tsne_image = tsne_norm.reshape(H, W, 3)
    
    # Save the image
    img = Image.fromarray(tsne_image)
    img.save('plucker_tsne_image.png')
    print("t-SNE image saved as 'plucker_tsne_image.png'")

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

    src_image_path = os.path.join(base_dir, folder_type, filename)
    tgt_image_path = os.path.join(base_dir, folder_type, filename2)
    src_image = load_resize_image_cv2(src_image_path).to(device)  # Add batch dimension
    tgt_image = load_resize_image_cv2(tgt_image_path).to(device)  # Add batch dimension

    # repo = 'isl-org/ZoeDepth'
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

    scene_id = '0a5c013435'
    focal_length_x = 1432.3682 /20.0
    focal_length_y = 1432.3682 /20.0
    principal_point_x = 954.08276 /10.0
    principal_point_y = 724.18256 /10.0

    # Intrinsic matrices
    src_intrinsic = np.array([[focal_length_x, 0, principal_point_x],
                            [0, focal_length_y, principal_point_y],
                            [0, 0, 1]])

    focal_length_x2 = 1431.4313 /20.0
    focal_length_y2 = 1431.4313 /20.0
    principal_point_x2 = 954.7021 /10.0
    principal_point_y2 = 723.6698 /10.0

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
    # src_homo_mat_sample = np.linalg.inv(src_homo_mat_sample)

    # tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ############aligned pose
    #                             [-0.1672538, -0.8868709, 0.43068102, 1.642482],
    #                             [0.966491, -0.23377006, -0.106051974, -1.1564484],
    #                             [0.0, 0.0, 0.0, 0.9999995]])
    tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ########raw target pose
            [-0.1672538, -0.8868709, 0.43068102, 1.642482],
            [0.966491, -0.23377006, -0.106051974, -1.1564484],
            [0.0, 0.0, 0.0, 0.9999995]])
    # tar_homo_mat_sample = np.linalg.inv(tar_homo_mat_sample)

    src_homo_mat = torch.tensor([src_homo_mat_sample], dtype=torch.float32, device=device).squeeze(0)
    tar_homo_mat = torch.tensor([tar_homo_mat_sample], dtype=torch.float32, device=device).squeeze(0)

    src_intrinsic = torch.tensor([src_intrinsic], dtype=torch.float32, device=device).squeeze(0)
    tar_intrinsic = torch.tensor([tar_intrinsic], dtype=torch.float32, device=device).squeeze(0)
    src_trans = torch.tensor([src_trans], dtype=torch.float32, device=device).view(-1, 3, 1)
    tar_trans = torch.tensor([tar_trans], dtype=torch.float32, device=device).view(-1, 3, 1)
    src_quaterns = torch.tensor([src_quatern], dtype=torch.float32, device=device)
    tar_quaterns = torch.tensor([tar_quatern], dtype=torch.float32, device=device)
    print("###########!!!!!!!!!!!!!!!!!!!###############")
    print(src_trans.shape)
    print(tar_trans.shape)
    print(src_homo_mat.shape)
    print(src_image.shape)
    
    #############notation!!!!!!!!!!!!!##########################
    ## src camera pose set to identity , origin as 0, so the taget camera pose with respect to the first canonical camera pose###########
    src_homo_canonical = torch.eye(4).to(device)
    # Compute rotation matrices from quaternions
    # src_rot_mat = torch.linalg.inv(quaternion_to_rotation_matrix(src_quaterns)).to(device)
    # tar_rot_mat = torch.linalg.inv(quaternion_to_rotation_matrix(tar_quaterns)).to(device)
    src_rot_mat = quaternion_to_rotation_matrix(src_quaterns).to(device)
    tar_rot_mat = quaternion_to_rotation_matrix(tar_quaterns).to(device)
    # Compute relative rotation matrix
    # relative_rot_mat = torch.bmm(tar_rot_mat, src_rot_mat.transpose(1, 2))
    # Compute relative homogeneous matrix
    relative_homo_mat = torch.matmul(tar_homo_mat, torch.linalg.inv(src_homo_mat))
    # relative_homo_mat = torch.matmul(tar_homo_mat, torch.inverse(src_homo_mat))

    # Image dimensions (for example purposes, assuming 800x600 resolution)
    _, H, W = src_image.shape

    print(tar_intrinsic.shape)
    # Compute Plücker coordinates
    #plucker_coordinates = compute_plucker_coordinates(src_homo_canonical, src_intrinsic, H, W) ######### source frmae #
    plucker_coordinates = compute_plucker_coordinates(relative_homo_mat, tar_intrinsic, H, W) ############ target frame ##########
    print(plucker_coordinates.shape)
    # Visualize Plücker coordinates using t-SNE and save as an RGB image
    visualize_plucker_coordinates_tsne(plucker_coordinates, H, W)
import torch
import torch.nn.functional as F

def fourier_features(x, L):
    """
    Compute the Fourier features for input x with scale L.
    Apply sin and cos to each dimension.
    """
    sin_features = torch.sin((2**L) * x)  # Apply sin to all dimensions
    cos_features = torch.cos((2**L) * x)  # Apply cos to all dimensions
    B = torch.cat([sin_features, cos_features], dim=-1)
    return B

def create_2d_coordinate_map(B, H, W):
    """
    Create a canonical 2D coordinate map normalized between -1 and 1 for a batch.
    """
    i, j = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij')
    coord_map = torch.stack([i, j], dim=-1)
    coord_map = coord_map.unsqueeze(0).repeat(B, 1, 1, 1)
    return coord_map

# def compute_fourier_feature_map(B, H, W, scales):
#     """
#     Create Fourier feature map for a batch of B HxW images with given scales.
#     """
#     coord_map = create_2d_coordinate_map(H, W)
#     coord_map = coord_map.reshape(-1, 2)  # Shape: (H*W, 2)
    
#     # Expand coord_map to batch size
#     coord_map = coord_map.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H*W, 2)
    
#     # Compute Fourier features for each scale and concatenate
#     fourier_maps = []
#     for L in scales:
#         fourier_map = fourier_features(coord_map, L)
#         fourier_maps.append(fourier_map)
        
#     # Concatenate the results along the last dimension
#     fourier_maps = torch.cat(fourier_maps, dim=-1)
    
    # Ensure the final shape is (B, H, W, 6)
    # assert fourier_maps.shape[-1] == len(scales) * 2, "Fourier feature dimension mismatch"
    # fourier_maps = fourier_maps.reshape(B, H, W, -1)
    
    # return fourier_maps

def compute_ray_directions(H, W, focal_length_x, focal_length_y):
    """
    Compute ray directions for all pixels in the image.
    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
    directions = torch.stack([(i - W * 0.5) / focal_length_x, -(j - H * 0.5) / focal_length_y, -torch.ones_like(i)], dim=-1)
    return directions

def get_view_directions(B, H, W, focal_length_x, focal_length_y, camera_pose):
    """
    Compute the view directions for a batch of images given their dimensions, focal length, and camera pose.
    """
    directions = compute_ray_directions(H, W, focal_length_x, focal_length_y)  # shape: (W, H, 3)
    directions = directions.reshape(-1, 3)
    
    # Apply rotation from camera pose
    directions = directions @ camera_pose[:3, :3].T  # Rotate ray directions by camera rotation (transpose for correct multiplication)
    
    # Normalize the directions to unit vectors
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    directions = directions.reshape(1, H, W, 3).repeat(B, 1, 1, 1)
    
    return directions

def compute_fourier_feature_map(B, H, W, scales, focal_length_x, focal_length_y, camera_pose):
    """
    Create Fourier feature map for a batch of HxW images with given scales.
    """
    coord_map = create_2d_coordinate_map(B, H, W)
    coord_map = coord_map.reshape(B, -1, 2)
    
    view_directions = get_view_directions(B, H, W, focal_length_x, focal_length_y, camera_pose)
    view_directions = view_directions.reshape(B, -1, 3)
    
    # Compute Fourier features for each scale and concatenate
    fourier_maps = []
    for L in scales:
        fourier_coord = fourier_features(coord_map, L)
        fourier_view = fourier_features(view_directions, L)
        print("######fourier !!!######")
        print(fourier_coord.shape)
        print(fourier_view.shape)
        combined_fourier = torch.cat([fourier_coord, fourier_view], dim=-1)
        fourier_maps.append(combined_fourier)
        
    # Concatenate the results along the last dimension
    fourier_maps = torch.cat(fourier_maps, dim=-1)
    
    # Ensure the final shape is (B, H, W, (2+3)*len(scales))
    fourier_maps = fourier_maps.reshape(B, H, W, -1)
    
    return fourier_maps



# # Example usage:
# B, H, W = 8, 32, 32  # Batch size, Height, Width
# scales = [1, 4, 7]  # Scales for Fourier features

# # Compute Fourier feature map
# fourier_map = compute_fourier_feature_map(B, H, W, scales)

# print("Fourier feature map shape:", fourier_map.shape)  # Should be (B, H, W, 6)
# Example usage:



B, H, W = 8, 32, 32
scales = [1, 4, 7]  # Scales for Fourier features
focal_length_x = 800.0  # Example focal length in pixels
focal_length_y = 800.0  # Example focal length in pixels

# Example camera pose (identity matrix for simplicity)
camera_pose = torch.eye(4)

# Compute Fourier feature map
fourier_map = compute_fourier_feature_map(B, H, W, scales, focal_length_x, focal_length_y, camera_pose)

print("Fourier feature map shape:", fourier_map.shape)  # Should be (B, H, W, (2+3)*len(scales))
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
# from dinov2.models.vision_transformer import vit_large 
from PIL import Image 
import torchvision.transforms as transforms 
import os
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import umap
import cv2
import matplotlib.pyplot as plt


# from dinov2.models import dinov2

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

######################################################################################
#                            epipolar line mask                                      #
#######################################################################################
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
    

def compute_skew_symmetric(v):   ###################tensor version########
    """
    Compute the skew-symmetric matrix from a 3D vector.
    
    Args:
        v (torch.Tensor): Tensor of shape (batch_size, 3)
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3, 3)
    """
    batch_size = v.size(0)
    M = torch.zeros((batch_size, 3, 3), device=v.device)
    M[:, 0, 1] = -v[:, 2]
    M[:, 0, 2] = v[:, 1]
    M[:, 1, 0] = v[:, 2]
    M[:, 1, 2] = -v[:, 0]
    M[:, 2, 0] = -v[:, 1]
    M[:, 2, 1] = v[:, 0]
    
    return M

def compute_fundamental_matrix(K1, K2, R, t):     ###############tensor version########
    """
    Compute the fundamental matrix from intrinsic matrices and relative pose.

    Args:
        K1 (torch.Tensor): Tensor of shape (batch_size, 3, 3). Source view intrinsic matrix.
        K2 (torch.Tensor): Tensor of shape (batch_size, 3, 3). Target view intrinsic matrix.
        R (torch.Tensor): Tensor of shape (batch_size, 3, 3). Relative rotation matrix from target to source view.
        t (torch.Tensor): Tensor of shape (batch_size, 3). Relative translation vector from target to source view.

    Returns:
        F (torch.Tensor): Tensor of shape (batch_size, 3, 3). Fundamental matrix.
    """
    batch_size = K1.size(0)
    print("############$$$$$$$$$$$$$$$$$$$##############")
    print(K1)
    print(K2)
    print(R)
    print(t)
    # Compute the essential matrix (using torch.bmm for batch matrix multiplication)
    t_skew = compute_skew_symmetric(t)
    # E = torch.bmm(K2.transpose(1, 2), torch.bmm(t_skew, torch.bmm(R, K1)))
    E = torch.bmm(t_skew, R)
    
    # Enforce the rank-2 constraint on the essential matrix
    U, S, Vt = torch.linalg.svd(E)
    S[:, 2] = 0
    E = torch.bmm(U, torch.bmm(torch.diag_embed(S), Vt))
    
    # Compute the fundamental matrix from the essential matrix (using torch.inverse)
    F = torch.bmm(torch.inverse(K2.transpose(1, 2)), torch.bmm(E, torch.inverse(K1)))
    
    return F

def calculate_epipolar_lines(points, fundamental_matrices):
    """
    Calculates the epipolar lines corresponding to points in the target view given the fundamental matrix.

    Args:
        points: A torch tensor of shape (batch_size, 3, N) representing the point coordinates (u, v, 1) in homogeneous coordinates in the target view.
        fundamental_matrices: A torch tensor of shape (batch_size, 3, 3) representing the fundamental matrices.

    Returns:
        A torch tensor of shape (batch_size, 3, N) representing the epipolar lines in homogeneous coordinates.
    """
    # Ensure points are in homogeneous coordinates and shape (batch_size, 3, N)
    assert points.shape[1] == 3, "Input points should have shape (batch_size, 3, N)"
    
    # Calculate the epipolar lines using the fundamental matrices (batch matrix multiplication)
    epipolar_lines = torch.bmm(fundamental_matrices, points)
    
    # Normalize the epipolar lines
    epipolar_lines /= epipolar_lines[:, 2:3, :]
    
    return epipolar_lines

def visualize_attention_map(attention_map, batch_idx=0, column_idx=600, save_path='attention_map_visualization.png'):
    """
    Visualizes and saves an attention map tensor as an image.
    
    Args:
        attention_map (torch.Tensor): The attention map tensor with shape [batch_size, height, width].
        batch_idx (int): Index of the batch to visualize.
        row_idx (int): Index of the row to visualize in the attention map.
        save_path (str): Path to save the image.
    """
    # Ensure the batch and row indices are within range
    # if batch_idx >= attention_map.size(0):
    #     raise ValueError("batch_idx is out of range.")
    # if row_idx >= attention_map.size(1):
    #     raise ValueError("row_idx is out of range.")

    # Select the batch and row
    print("########$$$$$$$$attention map$$$$$$$$$$$$#########")
    print(attention_map.shape)

    # attention_map = attention_map.permute(0, 2, 1)
    single_attention_map_batch_row = attention_map[batch_idx, :, column_idx]
    # Step 1: Reshape the tensor to (1024, 32, 32)
    reshaped_tensor = attention_map[0, :, :].view(1024, 32, 32)
    # Step 2: Rearrange these 32x32 images into a single 32x32 layout
    grid_size = 32
    grid_image = reshaped_tensor.view(grid_size, grid_size, 32, 32)
    grid_image = grid_image.permute(0, 2, 1, 3).contiguous()
    grid_image = grid_image.view(grid_size * 32, grid_size * 32)
    # Reshape the selected row to 32x32
    single_attention_map_reshaped = single_attention_map_batch_row.view(32, 32)

    # Normalize the tensor to the range [0, 255]
    attention_map_min = grid_image.min()
    attention_map_max = grid_image.max()
    attention_map_normalized = (grid_image - attention_map_min) / (attention_map_max - attention_map_min) * 255
    # Convert to numpy array and ensure the type is uint8
    attention_map_normalized = attention_map_normalized.cpu().numpy().astype(np.uint8)

    single_attention_map_min = single_attention_map_reshaped.min()
    single_attention_map_max = single_attention_map_reshaped.max()
    single_attention_map_normalized = (single_attention_map_reshaped - single_attention_map_min) / (single_attention_map_max - single_attention_map_min) * 255
    single_attention_map_normalized = single_attention_map_normalized.cpu().numpy().astype(np.uint8)

    # Save the tensor as an image
    image = Image.fromarray(attention_map_normalized)
    image.save(save_path)

    single_image = Image.fromarray(single_attention_map_normalized)
    new_path ='single_'+ save_path
    single_image.save(new_path)
    # Optionally, display the image using matplotlib
    # plt.imshow(attention_map_normalized, cmap='gray')
    # plt.title(f'Attention Map Visualization (Batch {batch_idx}, Row {row_idx})')
    # plt.axis('off')
    # plt.show()

class PatchifyAttention(nn.Module):
    def __init__(self, patch_size=16):
        super(PatchifyAttention, self).__init__()
        self.patch_size = patch_size
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 1024, 1024)
        B, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Height and Width must be divisible by patch_size."

        # Apply average pooling to get the average value for each patch
        x = self.avg_pool(x)
        
        # Flatten and reshape to (batch_size, num_patches, 1)
        x = x.view(B, -1, 1)        
        return x

###################epipolar attention###################################
class EpipolarAttention(nn.Module):
    def __init__(self, feature_dim, img_height, img_width, patch_size=2):
        super(EpipolarAttention, self).__init__()
        self.feature_dim = feature_dim
        self.img_height = img_height
        self.img_width = img_width
        self.sigmoid = nn.Softmax(dim=-1)     # nn.Sigmoid()
        self.patchify_attention_mask = PatchifyAttention(patch_size=patch_size)
    
    def forward(self, f_tar, f_src, K1, K2, R, t): ############# f_tar,  f_src is b x 3 x 1 as img pixel location, 
        ########## K in b x 3 x3, R in b x 3 x3, t b x 3 x1 
        # Compute the cross-view attention
        print("################input tensor shape##############")
        print(f_src.shape)
        print(f_tar.shape)
        self.f_src_flat = f_src.view(-1, self.feature_dim, self.img_height * self.img_width)  # Flatten the spatial dimensions
        self.f_tar_flat = f_tar.view(-1, self.feature_dim, self.img_height * self.img_width)    # Flatten the spatial dimensions
        # A_ij = torch.matmul(f_tar_flat.permute(0, 2, 1), f_src_flat.permute(0, 2, ))  # Compute affinity matrix (batch_size  feat_dim token_num)
        A = torch.einsum('bik,bkj->bij',  self.f_src_flat.permute(0, 2, 1), self.f_tar_flat)
        A = A.view(-1, self.img_height * self.img_width, self.img_height * self.img_width)  # Reshape affinity matrix
        batch_size =self.f_src_flat.size(0)
        
        # Compute the projeciting point and epipole points
        print("################tar proj tensor shape##############")
        print(self.f_src_flat.permute(0, 2, 1).shape)
        # print(K.shape)
        # print(R.shape)
        # print(self.f_tar_flat.shape)
        # print(t.shape)
        # Generate the index grid for the 32x32 grid
        index_x, index_y = torch.meshgrid(torch.arange(self.img_height, device = 'cuda', dtype=torch.float32),    \
                                          torch.arange(self.img_width, device = 'cuda', dtype=torch.float32), indexing='ij')
        index_x = index_x.reshape(-1)  # Flatten to a single dimension
        index_y = index_y.reshape(-1)  # Flatten to a single dimension

        # Create a combined index tensor of shape [1024, 3]
        combined_indices = torch.stack((index_x, index_y, torch.ones_like(index_x,  device='cuda', dtype=torch.float32)), dim=1)  # Shape [1024, 3]

        # Repeat the combined index tensor for each batch element
        combined_indices = combined_indices.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1024, 3]
        # combined_indices = combined_indices.view(-1, self.img_width, self.img_height, 3)

        combined_indices = combined_indices.permute(0, 2, 1)  # Shape [8, 3, 1024]
        print("################image index shape##############")
        print(combined_indices.shape)
        # print(K.shape)

        F_Matrix = compute_fundamental_matrix(K1, K2, R, t)
        print("################F matrix $$$$$$$$$$##############")
        print(F_Matrix)
        # epipolar_lines = torch.mv(F_Matrix[0, :, :], combined_indices[0, :, 512])
        # # Normalize the epipolar lines

        # # Compute the line endpoints based on the image dimensions
        # # x0, y0 = 0, int(-c / b)
        # # x1, y1 = width, int(-(c + a * width) / b)
        # # Compute the line endpoints based on the image dimensions
        # x0, y0 = [0, int(-epipolar_lines[2] / epipolar_lines[1])]
        # x1, y1 = [self.img_width, int(-(epipolar_lines[2] + epipolar_lines[0] * self.img_width) / epipolar_lines[1])]

        # Calculate epipolar lines
        epipolar_lines = torch.bmm(F_Matrix, combined_indices)  # Shape: (batch_size, 3, 1024)
        
        # Normalize the epipolar lines by the third element
        epipolar_lines = epipolar_lines / epipolar_lines[:, 2:3, :]
        
        # Compute the line endpoints based on the image dimensions
        x0 = torch.zeros(batch_size, 1, 1024, device=epipolar_lines.device)
        y0 = -epipolar_lines[:, 2:3, :] / epipolar_lines[:, 1:2, :]
        point_0 = torch.cat((x0, y0, torch.ones(batch_size, 1, 1024, device=epipolar_lines.device)), dim=1)  # Shape: (batch_size, 3, 1024)
        
        x1 = torch.full((batch_size, 1, 1024), self.img_width, device=epipolar_lines.device)
        y1 = -(epipolar_lines[:, 2:3, :] + epipolar_lines[:, 0:1, :] * self.img_width) / epipolar_lines[:, 1:2, :]
        point_1 = torch.cat((x1, y1, torch.ones(batch_size, 1, 1024, device=epipolar_lines.device)), dim=1)  # Shape: (batch_size, 3, 1024)

        print(epipolar_lines.shape)
        # tar_proj = torch.bmm(K2, (torch.bmm(R, (torch.bmm(torch.inverse(K1), combined_indices))) + t.view(-1, 3, 1).repeat(1, 1, self.img_height * self.img_width)))   #.squeeze(-1),  Compute the epipolar line parameters

        # origin = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32).view(3, 1).unsqueeze(0).repeat(batch_size, 1, self.img_height*self.img_width)  # Reshape affinity matrix
        # # o_proj = torch.bmm(K, (torch.bmm(R, (torch.bmm(torch.inverse(K), origin))) + t.view(-1, 3, 1).repeat(1, 1, self.img_height * self.img_width)))    #.squeeze(-1)
        # # origin = origin.permute(0, 2, 1)        
        # print(tar_proj)
        # print(origin.shape)
        
        # o_proj = torch.bmm(K2, torch.bmm(R, torch.bmm(torch.inverse(K1), origin)) +  t.view(-1, 3, 1).repeat(1, 1, self.img_height * self.img_width).view(-1, 3, self.img_height * self.img_width))
        # print(o_proj)
        # print(o_proj.shape)
        # # print(o_proj)  
        # # print(tar_proj)      
        # #  p_i_t_j = torch.matmul(R_t_i, torch.inverse(K) @ p_target.unsqueeze(-1)) + t_t_i.unsqueeze(-1)  # Equation (8)
        
        # # Compute the epipolar lines based on the projected points and origin
        # c = torch.linspace(-1, 1, steps=self.img_width)
        # print(c.shape)
        # epipolar_lines = o_proj + (tar_proj - o_proj)   ########## b x 3 x (h x w)
        ########## it can be replaced by own epipolar implementation

        d_epipolar = self.compute_epipolar_distance(combined_indices, point_0, point_1) #### b x N x N
        epipolar_line_thre = 0.10

        weight_map = 1 - self.sigmoid(5 * (d_epipolar - epipolar_line_thre))
        # self.compute_weight_map(f_src_flat, d_epipolar, epipolar_line_thre)
        
        # Apply epipolar attention
        A_weighted = weight_map  # A not used here, because targetfeature map will not be known Weight the affinity matrix
        # A_weighted = A * weight_map  # Weight the affinity matrix
        # A_weighted_flat = A_weighted.view(-1, self.img_height * self.img_width, self.img_height * self.img_width)


        attention_map = F.softmax(A_weighted, dim=1) ####attention over the row
        #visualization map check
        visualize_attention_map(attention_map)
        # Compute the output of the epipolar attention layer
        f_src_attended = torch.einsum('bik,bkj->bij',  attention_map, self.f_src_flat.permute(0, 2, 1))
        print("##############fusion attention output after epipolar attention map##########")
        print(attention_map.shape)
        print(f_src_attended.shape)
        # f_src_attended = torch.matmul(f_src_flat, attention_map)  # Apply the attention map
        f_src_attended = f_src_attended.view(-1, self.feature_dim, self.img_height, self.img_width)
        
        return f_src_attended

    def compute_epipolar_distance(self, f_src_flat, tar_proj, o_proj):
        # Compute the distance d(p_epipolar, p^j) according to the provided formula
        # f_src_flat, tar_proj, and o_proj are all of shape (b x 3 x N)

        # Compute the difference between target projections and origin projections
        diff = tar_proj - o_proj  # Shape: b x 3 x N

        # Expand f_src_flat and o_proj to enable broadcasting for pairwise column-wise cross products
        f_src_flat_expanded = f_src_flat.unsqueeze(3)  # Shape: b x 3 x N x 1
        o_proj_expanded = o_proj.unsqueeze(3)  # Shape: b x 3 x N x 1

        # Compute the cross product between each column of f_src_flat and each column of o_proj
        print("#######compute distance#############")
        print(f_src_flat_expanded.shape)
        print(o_proj_expanded.shape)
        print(diff.shape)
        cross_prod = torch.cross(f_src_flat_expanded - o_proj_expanded, diff.unsqueeze(2), dim=1)  # Shape: b x 3 x N x N

        # Compute the norm of the cross product and the norm of the difference
        cross_prod_norm = torch.norm(cross_prod, dim=1)  # Shape: b x N x N
        diff_norm = torch.norm(diff, dim=1, keepdim=True)  # Shape: b x 1 x N

        # Compute the distance d_epipolar
        d_epipolar = cross_prod_norm / diff_norm  # Shape: b x N x N
        print("#####epipolar distance output calcualted here !#####")
        print(d_epipolar.shape)
        return d_epipolar #### b x h x w
    


    # def epipolar_line_computation(p_target, K, R, t):
    #     """
    #     Compute the epipolar line on the source view image plane.

    #     Args:
    #         p_target (torch.Tensor): Point on the target view image plane (2D or 3D).
    #         K (torch.Tensor): Camera intrinsic matrix.
    #         R (torch.Tensor): Relative rotation matrix from target to source view.
    #         t (torch.Tensor): Relative translation vector from target to source view.

    #     Returns:
    #         torch.Tensor: Epipolar line parameters on the source view image plane.
    #     """
    #     p_i_j = K @ torch.matmul(R, torch.inverse(K) @ p_target.unsqueeze(-1)) + t.unsqueeze(-1)  # Equation (8)
    #     p_i_j = p_i_j.squeeze(-1)

    #     o_target = torch.tensor([0., 0., 0.])  # Camera origin at target view
    #     o_i_j = K @ torch.matmul(R, torch.inverse(K) @ o_target.unsqueeze(-1)) + t.unsqueeze(-1)  # Equation (9)
    #     o_i_j = o_i_j.squeeze(-1)

    #     epipolar_line_params = torch.stack([o_i_j, p_i_j - o_i_j], dim=-1)  # Line parameters: origin, direction

    #     return epipolar_line_params

##################################################################################
#                        ray map implementation from google                      #
##################################################################################
def compute_ray_directions(H, W, focal_length_x , focal_length_y):
    """
    Compute ray directions for all pixels in the image.
    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
    directions = torch.stack([(i - W * 0.5) / focal_length_x, -(j - H * 0.5) / focal_length_y, -torch.ones_like(i)], dim=-1)
    return directions

def compute_raymap(H, W, focal_length_x, focal_length_y, camera_pose):
    """
    Compute the raymap for an image given its dimensions, focal length, and camera pose.
    """
    directions = compute_ray_directions(H, W, focal_length_x, focal_length_y)  # shape: (W, H, 3)
    directions = directions.reshape(-1, 3)
    directions = directions @ camera_pose[:3, :3].T  # Rotate ray directions by camera rotation (transpose for correct multiplication)

    origins = camera_pose[:3, 3].expand_as(directions)  # Use camera translation as origin

    raymap = torch.cat([origins, directions], dim=-1)  # Concatenate origins and directions
    raymap = raymap.reshape(H, W, 6)  # Reshape to (H, W, 6)
    
    return raymap

def concatenate_raymap(latents, raymap):
    """
    Concatenate raymap with latent representations channel-wise.
    """
    B, C, H, W = latents.size()
    raymap = raymap.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # Shape: (B, 6, H, W)
    latents_with_raymap = torch.cat([latents, raymap], dim=1)  # Concatenate along channels
    return latents_with_raymap



#################################################################################
#                                 Core DiT Model                                #
#################################################################################
def exists(val):
    return val is not None

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, dropout=0.2):
        super().__init__()
        assert query_dim % heads == 0, 'dim should be divisible by num_heads'
        assert context_dim % heads == 0, 'dim should be divisible by num_heads'
        assert context_dim == query_dim, 'dim should be consistent for query and key token dimension'
        inner_dim = dim_head * heads
        context_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.norm_q = nn.LayerNorm(query_dim, eps=1e-6)  # Use LayerNorm for query normalization
        self.norm_k = nn.LayerNorm(context_dim, eps=1e-6)  # Use LayerNorm for query normalization
        self.linear = nn.Linear(context_dim, context_dim * 2, bias=False)  # Double the dimension
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        print("##### Cross Attention Info ################")
        # print(f"Context Dim: {context_dim}")
        print(f"Inner Dim: {inner_dim}")
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        print("##### Attention tokens start################")
        print(f"q Dim: {x.shape}")
        print(f"k Dim: {context.shape}")
        # Normalize query tokens using LayerNorm
        q = self.norm_q(x)
        # Project key and value tokens
        projected = self.linear(context)
        # k = context
        # v = context
        k, v = torch.split(projected, projected.size(2) // 2, dim=2)
        k = self.norm_k(k)

        # Rearrange tensors for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Calculate attention scores, apply scaling, and normalize with softmax
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if mask is not None:
            print("########mask is used!!!!!")
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)

        # Apply dropout on attention weights
        attn = self.attn_drop(attn)

        # Contextualize using attention weights and value vectors
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        ########Dino tokens cross attention############
        self.cross_atten = CrossAttention(query_dim = hidden_size, context_dim = hidden_size, heads = num_heads, \
                                           dim_head = hidden_size // num_heads, dropout=0.2)

    def forward(self, x, c, dino_feat):
        if dino_feat == None:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1) 
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1) 
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mca.unsqueeze(1) * self.cross_atten(modulate(self.norm3(x), shift_mca, scale_mca), dino_feat)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        dino_feat_size = 768, 
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.counter = 0 
        self.depth = depth
        # self.dino_feat = nn.Parameter(torch.zeros())     
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.dino_embedder = PatchEmbed(input_size, patch_size, dino_feat_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # self.semantic_model = vit_large(
        #     patch_size=14,
        #     img_size=526,
        #     init_values=1.0,
        #     block_chunks=0
        # )
        # self.semantic_model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        # Initialize dino feature patch_embed like nn.Linear (instead of nn.Conv2d):
        w2 = self.dino_embedder.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.dino_embedder.proj.bias, 0)
        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, dino_feat, y ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images) ####(1, 4 32, 32) for example
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t ####+ y  #####no class label emebedding                               # (N, D)
        dino_tokens = self.dino_embedder(dino_feat)
        for block in self.blocks:
            self.counter = self.counter + 1
            if self.counter == 14 or self.counter == 16:     #### cross attention only applied to specific layers
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, dino_tokens)       # (N, T, D)
            else: 
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, None)       # (N, T, D)
        self.counter = 0
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
    
def save_tensor_as_image(tensor, filename):
    # Normalize the tensor to [0, 1] range
    tensor = tensor.squeeze(0)  # Remove the batch dimension
    tensor = tensor.permute(1, 2, 0)  # Rearrange channels to (H, W, C)
    tensor_min = -10.0 #tensor.min()
    tensor_max = 10.0 #tensor.max()


    # Convert the flattened tensor to a NumPy array
    numpy_tensor = tensor.detach().cpu().numpy()
    flattened_data = numpy_tensor.reshape((-1, numpy_tensor.shape[2])) #######flatterned

    # Apply t-SNE (adjust hyperparameters as needed)
    # tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    # embedded_data = tsne.fit_transform(flattened_data)
    reducer = umap.UMAP(n_components=3)  # Reduce to 4 components
    embedded_data = reducer.fit_transform(flattened_data)
    # Apply t-SNE
    print('#######tsne used########')
    print(embedded_data.shape)

    # Reshape for color image (assuming 3 channels for RGB)
    if embedded_data.shape[1] < 2:
        raise ValueError("Embedded data requires at least 3 dimensions (RGB) for color image")

    # Convert the reduced NumPy array back to a PyTorch tensor
    tensor = torch.from_numpy(embedded_data.T).view(-1, 32, 32).to(device)  # Shape: (1, 3/756, 32, 32)
    # tensor = tensor.unsqueeze(0)
    print('#####12333344####')
    print(tensor.shape)
    print(tensor)
    # Transfer the tensor to CPU memory
    # cpu_array = tensor.cpu().numpy()
    # np.save('dino_frame_000440.npy', cpu_array)

    # # Convert the reduced tensor to a PIL Image
    reduced_tensor = tensor.squeeze(0).permute(1, 2, 0)  # Shape: (32, 32, 3)
    reduced_tensor = reduced_tensor.clamp(0, 1)  # Clamp values to [0, 1] range
    
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # Normalize to [0, 1]
    print("######save tensor to iamage for viz##########")
    print(tensor.shape)
    # Scale the tensor to [0, 255] range and convert to uint8
    tensor = tensor.mul(255).clamp(0, 255).byte()  # Multiply by 255 and clamp to [0, 255]
    if tensor.shape[2] > 4:
        tensor = tensor[:, : , 1:-1]
    # Resize the tensor to 256x256
    transform = transforms.Resize((256, 256), antialias=True)
    tensor = transform(tensor)

    # Convert the tensor to a PIL Image
    img = transforms.ToPILImage()(tensor)

    # Save the image
    img.save(filename)

# class CrossAttention(nn.Module):
#     def __init__(self, dim_q, dim_k, dim_v):
#         super().__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_k
#         self.dim_v = dim_v

#         self.q_proj = nn.Linear(dim_q, dim_q)
#         self.k_proj = nn.Linear(dim_k, dim_q)
#         self.v_proj = nn.Linear(dim_k, dim_v)
#         self.out_proj = nn.Linear(dim_v, dim_q)

#     def forward(self, x, y):
#         batch_size, _, height, width = x.shape

#         q = self.q_proj(x.view(batch_size, self.dim_q, -1)).transpose(1, 2)  # (batch_size, tokens, dim_q)
#         k = self.k_proj(y.view(batch_size, self.dim_k, -1)).transpose(1, 2)  # (batch_size, tokens, dim_q)
#         v = self.v_proj(y.view(batch_size, self.dim_k, -1)).transpose(1, 2)  # (batch_size, tokens, dim_v)

#         attn_weights = torch.bmm(q, k.transpose(1, 2))  # (batch_size, tokens_q, tokens_k)
#         attn_weights = attn_weights / (self.dim_q ** 0.5)
#         attn_weights = attn_weights.softmax(dim=-1)

#         attended_v = torch.bmm(attn_weights, v)  # (batch_size, tokens_q, dim_v)
#         attended_v = attended_v.transpose(1, 2).reshape(batch_size, self.dim_v, height, width)  # (batch_size, dim_v, height, width)

#         output = self.out_proj(attended_v)  # (batch_size, dim_q, height, width)

#         return output

# class CrossAttention2(nn.Module):
#     def __init__(self, dim_q, dim_k, dim_v, num_heads=8):
#         super().__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.num_heads = num_heads
#         self.head_dim = dim_q // num_heads

#         self.q_proj = nn.Linear(dim_q, dim_q)
#         self.k_proj = nn.Linear(dim_k, dim_q)
#         self.v_proj = nn.Linear(dim_k, dim_v)
#         self.out_proj = nn.Linear(dim_v, dim_q)

#     def forward(self, x, y):
#         batch_size, _, height, width = x.shape

#         q = self.q_proj(x.view(batch_size, self.dim_q, -1)).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, tokens_q, head_dim)
#         k = self.k_proj(y.view(batch_size, self.dim_k, -1)).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, tokens_k, head_dim)
#         v = self.v_proj(y.view(batch_size, self.dim_k, -1)).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, tokens_k, head_dim)

#         q_cat_k = torch.cat([q, k], dim=2)  # (batch_size, num_heads, tokens_q + tokens_k, head_dim)
#         attn_weights = torch.matmul(q_cat_k, q_cat_k.transpose(2, 3)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, tokens_q + tokens_k, tokens_q + tokens_k)
#         attn_weights = attn_weights.softmax(dim=-1)

#         attended_v = torch.matmul(attn_weights[:, :, :q.size(2), q.size(2):], v)  # (batch_size, num_heads, tokens_q, head_dim)
#         attended_v = attended_v.transpose(1, 2).reshape(batch_size, -1, self.dim_v)  # (batch_size, tokens_q, dim_v)

#         output = self.out_proj(attended_v).view(batch_size, self.dim_q, height, width)  # (batch_size, dim_q, height, width)

#         return output
  
if __name__ == "__main__":
    # Download all DiT checkpoints

    # print(feats)
    # min_val = feats.min().item()
    # max_val = feats.max().item()
    # print(min_val)
    # print(max_val)
    # Read a PIL image 

    # ###############pre processing for dino features ############################
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
    print("###### main starts#########")
    # semantic_model = vit_large(
    #     patch_size=14,
    #     img_size=526,
    #     init_values=1.0,
    #     block_chunks=0
    # )
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

    # tar_feats = src_feats
    print('#######feature shape######')
    print(src_feats.shape)
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
    # t_rel = tar_trans - tar_rot_mat @ src_rot_mat.T @ src_trans
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
    relative_homo_mat_R = relative_homo_mat[:, :3, :3]

    # For relative_homo_mat[:3, 3]
    relative_homo_mat_T = relative_homo_mat[:, :3, 3]
    patch_size = 2
    epi_atten = EpipolarAttention(src_feats.size(1), src_feats.size(2), src_feats.size(3), patch_size).to(device)
    epi_atten(tar_feats, src_feats, src_intrinsic, tar_intrinsic, relative_homo_mat_R, relative_homo_mat_T)
    # epi_atten(target_feats, tar_proj, o_proj)

    # transform = transforms.PILToTensor() 
    # Convert the PIL image to Torch tensor 
    # img_tensor = preprocess(image).to(device).unsqueeze(0)
    # print(img_tensor.shape) 


    # ##########Dino feats##########
    # Dino = DINO().to(device)
    # dino_feats = Dino(img_tensor)
    # # print(dino_feats)
    # ###############pre processing for dino features ############################

    # model = DiT_models['DiT-XL/2'](
    #     input_size=256 // 8,
    #     num_classes=1000
    # )
    # t = torch.randint(0, 1000, (1,), device=device)

    # model = model.to(device)
    # model(feats, t, dino_feats, torch.zeros(1, dtype=int).to(device))

    #EpipolarAttention
    # Example usage
    # save_tensor_as_image(dino_feats, 'output_dino_image.png')
    # save_tensor_as_image(feats, 'output_image.png')
    # min_val = dino_feats.min().item()
    # max_val = dino_feats.max().item()
    # print(min_val)
    # print(max_val)
    # semantic_model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))

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
import einops as E
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import umap
from torch import nn, einsum
from einops import rearrange, repeat

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

#################################################################################
#                            epipolar line mask                                 #
#################################################################################
def quaternion_to_rotation_matrix(quaternion):
  """
  Convert a quaternion to a 3x3 rotation matrix.

  Args:
      quaternion (torch.Tensor): 1D tensor representing the quaternion in the order [qw, qx, qy, qz].

  Returns:
      rotation_matrix (torch.Tensor): 3x3 rotation matrix.
  """
  qw, qx, qy, qz = quaternion.unbind(dim=1)

  # Compute the elements of the rotation matrix
  r11 = 1 - 2 * (qy**2 + qz**2)
  r12 = 2 * (qx * qy - qw * qz)
  r13 = 2 * (qx * qz + qw * qy)
  r21 = 2 * (qx * qy + qw * qz)
  r22 = 1 - 2 * (qx**2 + qz**2)
  r23 = 2 * (qy * qz - qw * qx)
  r31 = 2 * (qx * qz - qw * qy)
  r32 = 2 * (qy * qz + qw * qx)
  r33 = 1 - 2 * (qx**2 + qy**2)

  # Create the rotation matrix
  rotation_matrix = torch.stack([
      [r11, r12, r13],
      [r21, r22, r23],
      [r31, r32, r33]], dim=1)

  return rotation_matrix


def compute_skew_symmetric(v):
  """
  Compute the skew-symmetric matrix from a 3D vector.

  Args:
      v (torch.Tensor): 3x1 vector.

  Returns:
      M (torch.Tensor): 3x3 skew-symmetric matrix.
  """
  M = torch.tensor([
      [0, -v[2], v[1]],
      [v[2], 0, -v[0]],
      [-v[1], v[0], 0]], dtype=torch.float)
  return M


def compute_fundamental_matrix(K1, K2, R, t):
  """
  Compute the fundamental matrix from intrinsic matrix and relative pose.

  Args:
      K1 (torch.Tensor): 3x3 intrinsic matrix. Source view
      K2 (torch.Tensor): 3x3 intrinsic matrix. Target view
      R (torch.Tensor): 3x3 relative rotation matrix from target to source view.
      t (torch.Tensor): 3x1 relative translation vector from target to source view.

  Returns:
      F (torch.Tensor): 3x3 fundamental matrix.
  """
  # Move tensors to CUDA if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  K1, K2, R, t = K1.to(device), K2.to(device), R.to(device), t.to(device)

  # Compute the essential matrix (using torch.bmm for batch matrix multiplication)
  E = torch.bmm(K2.transpose(1, 2), torch.bmm(compute_skew_symmetric(t), torch.bmm(R, K1)))

  # Enforce the rank-2 constraint on the essential matrix
  U, S, Vt = torch.linalg.svd(E)
  S[2] = 0
  E = torch.bmm(U, torch.diag(S)) @ Vt

  # Compute the fundamental matrix from the essential matrix (using torch.inverse)
  F = torch.bmm(torch.inverse(K2.transpose(1, 2)), torch.bmm(E, torch.inverse(K1)))

  return F.to(device)  # Move the result back to CPU for further processing

###################epipolar attention###################################
class EpipolarAttention(nn.Module):
    def __init__(self, feature_dim, img_height, img_width):
        super(EpipolarAttention, self).__init__()
        self.feature_dim = feature_dim
        self.img_height = img_height
        self.img_width = img_width
        self.sigmoid = nn.Softmax(dim=-1)     # nn.Sigmoid()
    
    def forward(self, f_tar, f_src, K, R, t): ############# f_tar,  f_src is b x 3 x 1 as img pixel location, 
        ########## K in b x 3 x3, R in b x 3 x3, t b x 3 x1 
        # Compute the cross-view attention
        print("################input tensor shape##############")
        print(f_src.shape)
        print(f_tar.shape)

        self.f_src_flat = f_src.view(-1, self.feature_dim, self.img_height * self.img_width)  # Flatten the spatial dimensions
        self.f_tar_flat = f_tar.view(-1, self.feature_dim, self.img_height * self.img_width)    # Flatten the spatial dimensions
        # A_ij = torch.matmul(f_tar_flat.permute(0, 2, 1), f_src_flat.permute(0, 2, 1))  # Compute affinity matrix (batch_size  feat_dim token_num)
        A = torch.einsum('bik,bkj->bij',  self.f_src_flat.permute(0, 2, 1), self.f_tar_flat)
        A = A.view(-1, self.img_height * self.img_width, self.img_height * self.img_width)  # Reshape affinity matrix
        
        # Compute the projeciting point and epipole points
        print("################tar proj tensor shape##############")
        print(K.shape)
        print(R.shape)
        print(self.f_tar_flat.shape)
        print(t.shape)
        # Generate the index grid for the 32x32 grid
        index_x, index_y = torch.meshgrid(torch.arange(self.img_height), torch.arange(self.img_width), indexing='ij')
        index_x = index_x.reshape(-1)  # Flatten to a single dimension
        index_y = index_y.reshape(-1)  # Flatten to a single dimension

        # Create a combined index tensor of shape [1024, 3]
        combined_indices = torch.stack((index_x, index_y, torch.ones_like(index_x)), dim=1)  # Shape [1024, 3]

        # Repeat the combined index tensor for each batch element
        combined_indices = combined_indices.unsqueeze(0).repeat(self.f_src_flat.size(0), 1, 1)  # Shape [8, 1024, 3]
        combined_indices = combined_indices.view(-1, self.img_width, self.img_height, 3)
        tar_proj = torch.bmm(K, (torch.bmm(R, (torch.bmm(torch.inverse(K), combined_indices))) + t.view(-1, 3, 1).repeat(1, 1, self.img_height * self.img_width)))   #.squeeze(-1) # Compute the epipolar line parameters

        origin = torch.tensor([0.0, 0.0, 0.0]).view(3, 1).unsqueeze(0).repeat(1, 1, self.img_height*self.img_width)  # Reshape affinity matrix
        o_proj = torch.bmm(K, (torch.bmm(R, (torch.bmm(torch.inverse(K), origin))) + t.view(-1, 3, 1).repeat(1, 1, self.img_height * self.img_width)))    #.squeeze(-1)
        #  p_i_t_j = torch.matmul(R_t_i, torch.inverse(K) @ p_target.unsqueeze(-1)) + t_t_i.unsqueeze(-1)  # Equation (8)
        
        # Compute the epipolar lines based on the projected points and origin
        c = torch.linspace(-1, 1, steps=self.img_width)
        epipolar_lines = o_proj + c * (tar_proj - o_proj)   ########## b x 3 x (h x w)
        ########## it can be replaced by own epipolar implementation

        d_epipolar = self.compute_epipolar_distance(self.f_src_flat, tar_proj, o_proj) #### b x N x N
        epipolar_line_thre = 0.5

        weight_map = 1 - self.sigmoid(50 * (d_epipolar - epipolar_line_thre))
        # self.compute_weight_map(f_src_flat, d_epipolar, epipolar_line_thre)
        
        # Apply epipolar attention
        A_weighted_flat = A * weight_map  # Weight the affinity matrix
        # A_weighted_flat = A_weighted.view(-1, self.img_height * self.img_width, self.img_height * self.img_width)


        attention_map = F.softmax(A_weighted_flat, dim=1) ####attention over the row
        
        # Compute the output of the epipolar attention layer
        f_src_attended = torch.einsum('bik,bkj->bij',  attention_map, self.f_src_flat.permute(0, 2, 1))
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
        cross_prod = torch.cross(f_src_flat_expanded - o_proj_expanded, diff.unsqueeze(2), dim=1)  # Shape: b x 3 x N x N

        # Compute the norm of the cross product and the norm of the difference
        cross_prod_norm = torch.norm(cross_prod, dim=1)  # Shape: b x N x N
        diff_norm = torch.norm(diff, dim=1, keepdim=True)  # Shape: b x 1 x N

        # Compute the distance d_epipolar
        d_epipolar = cross_prod_norm / diff_norm  # Shape: b x N x N
        return d_epipolar #### b x 1 x (h x w)
    
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

    # def compute_weight_map(self, f_src_flat, d_epipolar, epipolar_line_thre):
        # Compute the epipolar line for each pixel in the target view
        # x = torch.arange(self.img_width).float()
        # y = torch.arange(self.img_height).float()
        # X, Y = torch.meshgrid(x, y)
        # P_t = torch.stack([X.flatten(), Y.flatten(), torch.ones_like(X.flatten())], dim=0)
        
        # # Compute the epipolar line distance for each pixel
        # d = torch.matmul(E_src[:, :3], P_t)
        # d_norm = torch.norm(d[:, :2], dim=1)  # Normalize using the first two components (u, v)
        # distances = torch.abs(d[2, :] / d_norm)  # Compute the distance to the epipolar line
        
        # Equation 12: m_{p_i, K, R, t}(v_j) = 1 - sigmoid(50 * (d - 0.05))
        # weight_map = 1 - self.sigmoid(50 * (d_epipolar - epipolar_line_thre))
        
        # return weight_map.view(self.img_height, self.img_width)

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
    vae_features = 'frame_000440.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feats = torch.from_numpy(np.load(os.path.join(base_dir, folder_type, vae_features))).to(device)
    print(feats.shape)
    # print(feats)
    # min_val = feats.min().item()
    # max_val = feats.max().item()
    # print(min_val)
    # print(max_val)
    print('#############')
    # Read a PIL image 

    ###############pre processing for dino features ############################
    image = Image.open(os.path.join(base_dir, folder_type, filename)) 
    
    # Define a transform to convert PIL  image to a Torch tensor 
    transform = transforms.Compose([ 
        transforms.PILToTensor() 
    ])

    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=Image.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the batch size
    b = 8
    feature_dim = 4 
    img_height = 32 
    img_width = 32
    # Initialize random tensors
    K = torch.randn(b, 3, 3, device='cuda')
    R = torch.randn(b, 3, 3, device='cuda')
    t = torch.randn(b, 3, 1, device='cuda')
    f_src_flat = torch.randn(b, 32, 32, 4, device='cuda')
    tar_proj = torch.randn(b, 32, 32, 4, device='cuda')
    N = img_width*img_height  # Example value for N, you can change it as needed

    # # Initialize random tensors
    # f_src_flat = torch.randn(b, 3, N, device='cuda')
    # tar_proj = torch.randn(b, 3, N, device='cuda')

    print("########  main #########")
    print(f_src_flat.shape)
    print(tar_proj.shape)
    # o_proj = torch.randn(b, 3, N, device='cuda')


    epi_atten = EpipolarAttention(feature_dim, img_height, img_width)
    epi_atten(tar_proj, f_src_flat, K, R, t)
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

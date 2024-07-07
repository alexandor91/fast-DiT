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

def quaternion_to_rotation_matrix(quaternions):
    """
    Converts a batch of quaternions to rotation matrices.
    
    Args:
        quaternions (np.ndarray): Array of shape (batch_size, 4)
        
    Returns:
        np.ndarray: Array of shape (3, 3)
    """
    w, x, y, z = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def compute_relative_transform(src_pose, tar_pose):
    # Compute the relative transformation matrix
    relative_transform = np.dot(tar_pose, np.linalg.inv(src_pose))
    return relative_transform

def clean_feature_map(feature_map_flat):
    # Remove rows with NaNs or Infs
    valid_indices = ~np.any(np.isnan(feature_map_flat) | np.isinf(feature_map_flat), axis=1)
    cleaned_feature_map = feature_map_flat[valid_indices]
    return cleaned_feature_map, valid_indices

def visualize_feature_map(feature_map, output_path):
    # Reshape the feature map to [H*W, C]
    height, width, channels = feature_map.shape
    feature_map_flat = feature_map.reshape(-1, channels)

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
    feature_map_3d_image[valid_indices] = feature_map_cleaned

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

def depth_to_3d_points_with_colors(depth_map, intrinsic_matrix, image):
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    x = (j - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (i - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map

    points_3D = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = np.transpose(image, (1, 2, 0)).reshape(-1, 3)  # image warp (H*W, 3), efature map in Shape: (H*W, 4)
    
    return points_3D, colors

def transform_points_with_colors(points, colors, rotation_matrix, translation_vector):
    points_transformed = np.dot(points, rotation_matrix.T) + translation_vector
    points_with_colors = np.concatenate((points_transformed, colors), axis=1)  # Shape: (H*W, 7)
    return points_with_colors

def project_points_with_colors(points_with_colors, intrinsic_matrix):
    points = points_with_colors[..., :3]
    colors = points_with_colors[..., 3:]

    points_projected = np.dot(points, intrinsic_matrix.T)
    points_2d = points_projected[:, :2] / points_projected[:, 2:3]
    points_2d_with_colors = np.concatenate((points_2d, colors), axis=1)  # Shape: (H*W, 6)
    return points_2d_with_colors

def populate_image_with_colors(projected_points_2D_colors, H, W):
    image_height = H
    image_width = W

    # Initialize an empty image with 4 channels
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    # Get the x and y coordinates
    x_coords = projected_points_2D_colors[..., 0]
    y_coords = projected_points_2D_colors[..., 1]
    colors = projected_points_2D_colors[..., 2:]

    # Create a mask for valid coordinates
    valid_mask = (x_coords >= 0) & (x_coords < image_width) & (y_coords >= 0) & (y_coords < image_height)

    # Convert the coordinates to integer type
    x_coords = x_coords.astype(int)
    y_coords = y_coords.astype(int)

    # Use the valid mask to filter out-of-bound coordinates
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_colors = colors[valid_mask]

    image[valid_y, valid_x] = valid_colors
    return image

def warp_image(src_image, tgt_points):
    C, H, W = src_image.shape
    tgt_points = tgt_points.reshape(H, W, 2)  # Shape: (H, W, 2)
    
    tgt_points[..., 0] = (2.0 * tgt_points[..., 0] / (W - 1)) - 1.0
    tgt_points[..., 1] = (2.0 * tgt_points[..., 1] / (H - 1)) - 1.0

    # Use OpenCV to warp the image
    src_image = src_image.transpose(1, 2, 0)  # Convert to (H, W, C)
    map_x, map_y = tgt_points[..., 0], tgt_points[..., 1]
    map_x = (map_x + 1) * (W - 1) / 2
    map_y = (map_y + 1) * (H - 1) / 2

    warped_image = cv2.remap(src_image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return warped_image.transpose(2, 0, 1)  # Convert back to (C, H, W)


def center_crop_img_and_resize(src_image, image_size):
    """
    Center cropping implementation from ADM, modified to work with OpenCV images.
    """
    while min(src_image.shape[:2]) >= 2 * image_size:
        new_size = (src_image.shape[1] // 2, src_image.shape[0] // 2)
        src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_AREA)

    scale = image_size / min(src_image.shape[:2])
    print("here is scale!!!!")
    print(scale)
    new_size = (round(src_image.shape[1] * scale), round(src_image.shape[0] * scale))
    src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_CUBIC)

    crop_y = (src_image.shape[0] - image_size) // 2
    crop_x = (src_image.shape[1] - image_size) // 2
    cropped_image = src_image[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

    print(cropped_image.shape)
    return cropped_image


def load_resize_image_cv2(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a numpy array and change dtype to float32
    image = image.astype(np.float32)
    # Rearrange the dimensions to (C, H, W)
    # image = np.transpose(image, (2, 0, 1))
    return image

def calculate_valid_pixel_ratio(binary_masks):
    # The binary_masks array shape should be (h, w, 1)
    assert binary_masks.ndim == 3 and binary_masks.shape[-1] == 1, \
        "binary_masks should be of shape (h, w, 1)"
    
    # Remove the channel dimension
    binary_masks = binary_masks[:, :, 0]

    # Sum of valid pixels in the binary mask (binary_masks > 0)
    valid_pixel_count = np.sum(binary_masks > 0).astype(np.float32)
    
    # Total number of pixels in the image (h * w)
    total_pixels = binary_masks.shape[0] * binary_masks.shape[1]
    
    # Calculate the ratio
    valid_pixel_ratio = valid_pixel_count / total_pixels
    
    return valid_pixel_ratio

# Function to save depth map
def save_depth_map(depth_map, save_path):
    depth_map = np.squeeze(depth_map)
    plt.imsave(save_path, depth_map, cmap='inferno')

def project_and_save_tsne_image(src_features, output_path='tsne_src_viz.png'):
    """
    Projects the input features from 4 dimensions to 3 dimensions using t-SNE and saves it as an image.
    
    Args:
    - src_features (numpy array): The input array with shape (1, 4, 32, 32).
    - output_path (str): The path where the output image will be saved.
    """
    # Remove the batch dimension
    if src_features.ndim == 4:
       src_features = src_features[0]  # Shape: (4, 32, 32)

    # Reshape to (4, 1024)
    H, W = src_features.shape[1], src_features.shape[2]
    src_features_reshaped = src_features.reshape(4, -1).T  # Shape: (1024, 4)

    # Perform t-SNE to reduce to 3 dimensions
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(src_features_reshaped)  # Shape: (1024, 3)

    # Reshape back to image format (32, 32, 3)
    tsne_image = tsne_result.reshape(H, W, 3)

    # Normalize to range [0, 255]
    tsne_image = (tsne_image - tsne_image.min()) / (tsne_image.max() - tsne_image.min()) * 255
    tsne_image = tsne_image.astype(np.uint8)

    # Save the image using OpenCV
    cv2.imwrite(output_path, tsne_image)
    return tsne_image


if __name__ == "__main__":
    # Download all DiT checkpoints
    print("###### main starts#########")
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/scene0000_00/color'
    img_folder_type = 'rgb'
    filename = '0.jpg'
    src_vae_features = 'frame_000440.npy'

    filename2 = '70.jpg'
    tar_vae_features = 'frame_000470.npy'

    ###########load VAE numpy features################
    # src_feats = np.load(os.path.join(base_dir, folder_type, src_vae_features))
    # tar_feats = np.load(os.path.join(base_dir, folder_type, tar_vae_features))

    src_image_path = os.path.join(base_dir, folder_type, filename)
    tgt_image_path = os.path.join(base_dir, folder_type, filename2)
    src_image = load_resize_image_cv2(src_image_path)
    tgt_image = load_resize_image_cv2(tgt_image_path)
    # print("$$$$$$$$$$$$$$")
    # print(src_image.shape)
    # src_image = center_crop_img_and_resize(src_image, 256)
    # tgt_image = center_crop_img_and_resize(tgt_image, 256)

    # src_image = project_and_save_tsne_image(src_feats, 'tsne_src_viz.png')
    # tgt_image = project_and_save_tsne_image(tar_feats, 'tsne_tar_viz.png')

    print("#####cropped image size#######")
    # print(src_feats.shape)

    json_file = 'fast-DiT/data/scene0000_00/scene_data.json'
    with open(os.path.join(base_dir, json_file), 'r') as file:
        data = json.load(file)

    # Initialize variables for pose and intrinsics
    src_intrinsic = None
    tar_intrinsic = None
    src_homo_mat_sample = None
    tar_homo_mat_sample = None
    # Iterate through the data to find the specific timestamp
    # Check if the frame_id exists in the data
    # src_frame_id = "frame_000440"
    # tar_frame_id = "frame_000470"
    src_intrinsic = np.array(data["intrinsic"]["intrinsic_color"])[:3, :3]

    src_homo_mat_sample = np.array(data["poses"]["0"])

    tar_intrinsic = np.array(data["intrinsic"]["intrinsic_color"])[:3, :3]
    print(tar_intrinsic)
    # Extract the pose for frame 4510
    tar_homo_mat_sample = np.array(data["poses"]["70"])    
    #########downsampling to 32 x 32 feature map

    src_homo_mat_sample = np.linalg.inv(src_homo_mat_sample)
    tar_homo_mat_sample = np.linalg.inv(tar_homo_mat_sample)

    # scale1 = 32/min(src_image.shape[:2]) #########final feature map size is 32
    # scale2 = 32/(min(src_image.shape[:2]) * (src_intrinsic[1, 1]/src_intrinsic[0, 0]))
    # src_intrinsic[0, 0] = src_intrinsic[0, 0] * scale1
    # src_intrinsic[1, 1] = src_intrinsic[1, 1] * scale2
    # src_intrinsic[0, 2] = 128 /8 #954.7021
    # src_intrinsic[1, 2] = 128 /8 #723.6698

    # scale1 = 32/min(tgt_image.shape[:2]) #########final feature map size is 32
    # scale2 = 32/(min(tgt_image.shape[:2]) * (tar_intrinsic[1, 1]/tar_intrinsic[0, 0]))
    # tar_intrinsic[0, 0] = tar_intrinsic[0, 0] * scale1
    # tar_intrinsic[1, 1] = tar_intrinsic[1, 1] * scale2
    # tar_intrinsic[0, 2] = 128 /8 #954.7021
    # tar_intrinsic[1, 2] = 128 /8 #723.6698

    ####test inputs below  by manual init values from real data####################
    # scene_id = '0a5c013435'
    # focal_length_x = 1432.3682
    # focal_length_y = 1432.3682
    # scale1 = 32/min(src_image.shape[:2]) #########final feature map size is 32
    # scale2 = 32/min(src_image.shape[:2])
    # focal_length_x = focal_length_x * scale1
    # focal_length_y = focal_length_y * scale2

    # principal_point_x = 128 / 8 ### 256 square image size #954.08276
    # principal_point_y = 128 / 8 ### 256 square image size  #724.18256

    # src_intrinsic = np.array([[focal_length_x, 0, principal_point_x],
    #                         [0, focal_length_y, principal_point_y],
    #                         [0, 0, 1]])
 
    # focal_length_x2 = 1431.4313
    # focal_length_y2 = 1431.4313
    # focal_length_x2 = focal_length_x2 * scale1
    # focal_length_y2 = focal_length_y2 * scale2
    # principal_point_x2 = 128 /8 #954.7021
    # principal_point_y2 = 128 /8 #723.6698

    # tar_intrinsic = np.array([[focal_length_x2, 0, principal_point_x2],
    #                         [0, focal_length_y2, principal_point_y2],
    #                         [0, 0, 1]])

    # src_quatern = np.array([0.837752, 0.490157, -0.150019, 0.188181])
    # src_trans = np.array([0.158608, 1.22818, -1.60956])
    # tar_quatern = np.array([0.804066, 0.472639, -0.25357, 0.256501])
    # tar_trans = np.array([0.473911, 1.28311, -1.5215])

    # src_homo_mat_sample = np.array([[0.42891303, 0.40586197, 0.8070378, 1.4285464],
    #                                 [-0.06427293, -0.8774122, 0.4754123, 1.6330968],
    #                                 [0.9010566, -0.25578123, -0.35024756, -1.2047926],
    #                                 [0.0, 0.0, 0.0, 0.99999964]])
    # src_homo_mat_sample = np.linalg.inv(src_homo_mat_sample)

    # tar_homo_mat_sample = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839],
    #                                 [-0.1672538, -0.8868709, 0.43068102, 1.642482],
    #                                 [0.966491, -0.23377006, -0.106051974, -1.1564484],
    #                                 [0.0, 0.0, 0.0, 0.9999995]])
    # tar_homo_mat_sample = np.linalg.inv(tar_homo_mat_sample)

    # src_rot_mat = quaternion_to_rotation_matrix(src_quatern)
    # tar_rot_mat = quaternion_to_rotation_matrix(tar_quatern)

    # relative_rot_mat = np.dot(tar_rot_mat, src_rot_mat.T)
    relative_homo_mat = np.dot(tar_homo_mat_sample, np.linalg.inv(src_homo_mat_sample))
    depth_folder_type = 'fast-DiT/data/scene0000_00/depth'

    depth_file = '0.png'  # Replace with your depth map image path
    depth_map = cv2.imread(os.path.join(base_dir, depth_folder_type, depth_file), cv2.IMREAD_UNCHANGED)
    # depth_map = center_crop_img_and_resize(depth_map, 256)
    # fig, ax = plt.subplots()

    print(depth_map.shape)
    # # Display depth map with jet colormap
    # im = ax.imshow(depth_map, cmap='magma')

    # # Save the displayed image with jet colormap
    # plt.savefig('depth_map_magma_color.png')
    # plt.close(fig)

    if depth_map is None:
        raise ValueError(f"Failed to load depth map from {depth_file}")

    if depth_map.dtype != np.uint16:
        raise ValueError("Depth map is not in uint16 format")

    depth_map_float = depth_map.astype(np.float32)
    depth_map_float /= 1000.0

    H, W, C = src_image.shape
    factor = H / depth_map_float.shape[0]
    # factor2 = W / depth_map_float.shape[1]
    # assert factor1 == factor2, f"Factors are not equal: factor1 = {factor1}, factor2 = {factor2}"
    # factor = factor2
    new_H = int(H / factor)
    new_W = int(W / factor)

    # downsampled_image = cv2.resize(np.transpose(src_image, (1, 2, 0)), (new_W, new_H))
    depth_map_float = cv2.resize(depth_map_float, (W, H), interpolation=cv2.INTER_CUBIC)  ### cv2.INTER_LINEAR
    # src_feats = src_feats[0]
    # tar_feats = tar_feats[0]    
    downsampled_image = np.transpose(src_image, (2, 0, 1))

    points_3D, colors = depth_to_3d_points_with_colors(depth_map_float, src_intrinsic, downsampled_image)
    print("###colors shape#####")
    print(colors.shape)
    # Transform points with colors to target camera frame
    transformed_with_colors = transform_points_with_colors(points_3D, colors, relative_homo_mat[:3, :3], relative_homo_mat[:3, 3])

    # Ensure the points and colors arrays have compatible shapes before concatenation
    # transformed_with_colors = np.concatenate((transformed_points, colors), axis=1)
    print(transformed_with_colors.shape)

    # Project transformed 3D points with colors onto the target camera image plane
    projected_points_2D_colors = project_points_with_colors(transformed_with_colors, tar_intrinsic)

    # Warp the source image to the target view
    warped_image = populate_image_with_colors(projected_points_2D_colors, H, W)
    print(warped_image.shape)
    # project_and_save_tsne_image(warped_image, 'tsne_warped_viz.png')
    # Save the warped image
    cv2.imwrite('Warped-src-img.png', warped_image)

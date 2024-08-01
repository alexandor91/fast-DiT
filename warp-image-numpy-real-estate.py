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

    points_3D = np.stack((x, y, z), axis=-1)  ###.reshape(-1, 3)
    colors = np.transpose(image, (1, 2, 0))    #  .reshape(-1, 3)
    
    return points_3D, colors

def transform_points_with_colors(points, colors, rotation_matrix, translation_vector):
    points_transformed = np.dot(points, rotation_matrix.T) + translation_vector
    points_with_colors = np.concatenate((points_transformed, colors), axis=1)  # Shape: (H*W, 7)
    return points_with_colors

def depth_to_3d_points_with_colors(depth_map, intrinsic_matrix, image):
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    x = (j - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (i - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map

    points_3D = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = np.transpose(image, (1, 2, 0)).reshape(-1, 3)
    
    return points_3D, colors

def project_points_with_colors(points_with_colors, intrinsic_matrix):
    points = points_with_colors[..., :3]
    colors = points_with_colors[..., 3:]

    points_projected = np.dot(points, intrinsic_matrix.T)
    points_2d = points_projected[:, :2] / points_projected[:, 2:3]
    points_2d_with_colors = np.concatenate((points_2d, colors), axis=1)
    return points_2d_with_colors


def populate_image_with_colors(projected_points_2D_colors, H, W):
    image_height = H
    image_width = W

    print("############populate the colors")
    print(projected_points_2D_colors.shape)

    # Initialize an empty image
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    # # Get the x and y coordinates
    x_coords = projected_points_2D_colors[..., 0]
    y_coords = projected_points_2D_colors[..., 1]
    colors = projected_points_2D_colors[..., 2:]
    # # Create a mask for valid coordinates
    valid_mask = (x_coords >= 0) & (x_coords < image_width) & (y_coords >= 0) & (y_coords < image_height)

    # # Convert the coordinates to integer type
    x_coords = x_coords.astype(int)
    y_coords = y_coords.astype(int)

    # # Use the valid mask to filter out-of-bound coordinates
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_colors = colors[valid_mask]
    print(valid_colors.shape)
    print(valid_x.shape)
    print(colors.shape)
    image[valid_y, valid_x] = valid_colors
    # print(image.shape)

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
    folder_type = 'fast-DiT/data'
    folder_type = 'fast-DiT/data/real-estate/rgb'
    img_folder_type = 'rgb'
    filename = 'frame_000440.jpg'
    filename = '86352933.png'
    src_vae_features = 'frame_000440.npy'

    filename2 = 'frame_000470.jpg'
    filename2 = '87654233.png'
    tar_vae_features = 'frame_000470.npy'

    folder_type = 'fast-DiT/data/realestate/1'
    file_type = 'rgb'
    output_folder = 'warped-output'
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
    filenames.sort()
    print(filenames)

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
    for filename in filenames:          
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
        downsampled_image = np.transpose(src_image, (2, 0, 1))

        points_3D, colors = depth_to_3d_points_with_colors(depth_map_float, src_intrinsic, downsampled_image)
        print("###colors shape#####")
        print(colors.shape)
        print(points_3D.shape)
        # Transform points with colors to target camera frame
        transformed_with_colors = transform_points_with_colors(points_3D, colors, relative_homo_mat[:3, :3], relative_homo_mat[:3, 3])

        # Ensure the points and colors arrays have compatible shapes before concatenation
        # transformed_with_colors = np.concatenate((transformed_points, colors), axis=1)
        print(transformed_with_colors.shape)

        # Project transformed 3D points with colors onto the target camera image plane
        projected_points_2D_colors = project_points_with_colors(transformed_with_colors, tar_intrinsic)

        # Warp the source image to the target view
        warped_image = populate_image_with_colors(projected_points_2D_colors, H, W)
        warped_image = warped_image.transpose(2, 0, 1)
        print(warped_image.shape)
        # Check if the directory exists, create if it doesn't
        output_dir = os.path.join(base_dir, folder_type, output_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define the full path for the file
        save_path = os.path.join(output_dir, str(frame_id) + '.png')
        
        # Save the numpy array as a PNG image
        cv2.imwrite(save_path, warped_image.transpose(1, 2, 0))
        # project_and_save_tsne_image(warped_image, 'tsne_warped_viz.png')

    # fig, ax = plt.subplots()

    # # Display depth map with jet colormap
    # im = ax.imshow(depth_map, cmap='magma')
    ## plt.colorbar(im, ax=ax)  # Add colorbar for reference

    # Save the displayed image with jet colormap
    # plt.savefig('depth_map_magma_color.png')
    # plt.close(fig)


    # depth_map_float /= 255.0
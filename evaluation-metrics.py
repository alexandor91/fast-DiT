import os
import numpy as np
import torch
import lpips
import cv2
from torchvision import models, transforms
from scipy.linalg import sqrtm
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel
import json

# Load the LPIPS model
lpips_model = lpips.LPIPS(net='alex')


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

# Helper functions
def load_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = transforms.ToTensor()(image)
    return image

def compute_fid(real_features, generated_features):
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    mu2 = np.mean(generated_features, axis=0)
    sigma2 = np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to compute the KID
def compute_kid(real_features, generated_features, degree=3, coef0=1, gamma=None):
    K_real = polynomial_kernel(real_features, degree=degree, coef0=coef0, gamma=gamma)
    K_gen = polynomial_kernel(generated_features, degree=degree, coef0=coef0, gamma=gamma)
    K_real_gen = polynomial_kernel(real_features, generated_features, degree=degree, coef0=coef0, gamma=gamma)
    
    m = real_features.shape[0]
    n = generated_features.shape[0]
    
    kid = (np.sum(K_real) / (m * m)) + (np.sum(K_gen) / (n * n)) - (2 * np.sum(K_real_gen) / (m * n))
    return kid

# Function to compute the Inception Score
def compute_is(generated_features):
    kl_div = generated_features * (np.log(generated_features) - np.log(np.expand_dims(np.mean(generated_features, 0), 0)))
    is_score = np.exp(np.mean(np.sum(kl_div, 1)))
    return is_score

def compute_lpips(img1, img2):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    return lpips_model(img1, img2).item()

def compute_psnr(img1, img2):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return psnr(img1, img2, data_range=img1.max() - img1.min())

def compute_ssim(img1, img2, win_size=3):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    win_size = 3
    # Ensure that the window size does not exceed the dimensions of the images
    win_size = min(win_size, img1.shape[0], img1.shape[1])
    
    # Compute SSIM
    return ssim(img1, img2, win_size=win_size, channel_axis=-1, data_range=img1.max() - img1.min())

def get_sift_matches(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return points1, points2, matches

def get_essential_matrix(pose1, pose2, intrinsics):
    pose1_inv = np.linalg.inv(pose1)
    rel_pose = np.dot(pose1_inv, pose2)
    
    T = rel_pose[:3, 3]
    R = rel_pose[:3, :3]
    
    T_cross = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])
    
    E = np.dot(R, T_cross)
    E = np.dot(intrinsics.T, np.dot(E, intrinsics))
    
    return E

# def get_min_dist(p1, E, kp):
#     p1 = np.append(p1, 1)
#     epipolar_line = np.dot(p1, E.T)
#     norm = np.linalg.norm(epipolar_line[:2])
#     epipolar_line /= norm
    
#     kp_h = np.append(kp, 1)
#     s = -np.dot(epipolar_line, kp_h) / norm
    
#     return np.abs(s)


def get_min_dist(p1, E, kp):

    """
    Calculate the distance from a point to the epipolar line.
    
    Args:
    p1 (np.array): 2D point in the first image (2,)
    E (np.array): Essential matrix (3, 3)
    kp (np.array): 2D point in the second image to compare against (2,)
    
    Returns:
    float: The distance from kp to the epipolar line
    """
    # Convert p1 to homogeneous coordinates
    p1_h = np.append(p1, 1)
    
    # Calculate the epipolar line l' = E^T * p1
    epipolar_line = np.dot(E.T, p1_h)
    
    # Normalize the line equation
    a, b, c = epipolar_line
    norm = np.sqrt(a**2 + b**2)
    epipolar_line /= norm
    
    # Convert kp to homogeneous coordinates
    kp_h = np.append(kp, 1)
    
    # Calculate the distance from point to line
    distance = np.abs(np.dot(epipolar_line, kp_h)) / np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
    return distance
    

def compute_tsed(img1, img2, pose1, pose2, src_intrinsics, tar_intrinsics, threshold=12.0):
    points1, points2, matches = get_sift_matches(img1, img2)
    
    if len(matches) == 0:
        return 0
    
    E12 = get_essential_matrix(pose1, pose2, src_intrinsics)
    E21 = get_essential_matrix(pose2, pose1, tar_intrinsics)
    
    seds = []
    for p1, p2 in zip(points1, points2):
        sed1 = get_min_dist(p1, E12, p2)
        sed2 = get_min_dist(p2, E21, p1)
        sed = 0.5 * (sed1 + sed2)
        seds.append(sed)
    # Convert seds to a numpy array if it's not already
    seds_array = np.array(seds)
    # Check which elements are less than the threshold
    below_threshold = seds_array < threshold
    # Count the number of True values (elements below the threshold)
    count = np.sum(below_threshold)

    n_matches = len(seds)
    # print("########total matches is !!!!!!")
    # print(n_matches)

    median_sed = np.median(seds_array) if n_matches > 0 else 1e8
    
    return count, median_sed

# Example usage in the evaluate function
def tsed_evaluate(generated_dir, poses, intrinsics):
    generated_files = sorted(os.listdir(generated_dir))
    # ground_truth_files = sorted(os.listdir(ground_truth_dir))
    
    tsed_scores = []
    
    for i in range(len(generated_files) - 1):
        gen_image1 = cv2.imread(os.path.join(generated_dir, generated_files[0]))
        gen_image2 = cv2.imread(os.path.join(generated_dir, generated_files[i + 1]))
        pose1 = poses[i]
        pose2 = poses[i + 1]
        # gt_image1 = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i]))
        # gt_image2 = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i + 1]))
        src_intrinsics = intrinsics[i]
        tar_intrinsics = intrinsics[i+1]
        
        count1, media_dist1 = compute_tsed(gen_image1, gen_image2, pose1, pose2, src_intrinsics, tar_intrinsics)
        # tsed_score_gt = compute_tsed(gt_image1, gt_image2, pose1, pose2, src_intrinsics, tar_intrinsics)
        tsed_scores.append((count1, media_dist1))
    
    avg_tsed_count = np.mean([score[0] for score in tsed_scores])
    avg_tsed_dis = np.mean([score[1] for score in tsed_scores])

    # avg_tsed_gt = np.mean([score[1] for score in tsed_scores])
    
    print(f'Average TSED (Generated): {avg_tsed_count}, {avg_tsed_dis}')


# Evaluation script
def evaluate(generated_dir, ground_truth_dir):
    generated_files = sorted(os.listdir(generated_dir))
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    
    fid_scores = []
    is_scores = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    tsed_scores = []
    
    # Load InceptionV3 model for feature extraction
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # Remove final classification layer
    inception_model.eval()
    
    real_features = []
    generated_features = []
    all_kid_values = []

    for gen_file, gt_file in tqdm(zip(generated_files, ground_truth_files), total=len(generated_files)):
        gen_image = load_image(os.path.join(generated_dir, gen_file))
        gt_image = load_image(os.path.join(ground_truth_dir, gt_file))
        
        # Compute features using InceptionV3
        with torch.no_grad():
            real_feature = inception_model(gt_image.unsqueeze(0)).flatten(1).cpu().numpy()
            gen_feature = inception_model(gen_image.unsqueeze(0)).flatten(1).cpu().numpy()
            print("#######$$$$$$ inception feathre $$$$$$$###########")
            print(real_feature)
            print(gen_feature)
        real_features.append(real_feature)
        generated_features.append(gen_feature)
        
        # Compute metrics
        lpips_score = compute_lpips(gen_image, gt_image)
        psnr_score = compute_psnr(gen_image, gt_image)
        ssim_score = compute_ssim(gen_image, gt_image, 3)
        print("#######$$$$$$$$$$$$$$#########")
        print(real_feature)
        print(gen_feature)
        # fid_score = compute_fid(real_features, generated_features)
        kid_value = compute_kid(real_feature, gen_feature)
        ###########
        # Compute TSED if necessary
        # tsed_score = compute_tsed(gen_image.numpy(), gt_image.numpy())
        
        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        all_kid_values.append(kid_value)
        # fid_scores.append(fid_score)
        # tsed_scores.append(tsed_score)

    # After the loop
    real_features = np.array(real_features)
    generated_features = np.array(generated_features)
    # Compute FID and IS
    real_features = np.vstack(real_features)
    generated_features = np.vstack(generated_features)

    fid_score = compute_fid(generated_features, real_features)
    # is_score = compute_is(generated_features)
    
    # print(f'Inception Score (IS): {is_score}')
    print(f'LPIPS: {np.mean(lpips_scores)}')
    print(f'PSNR: {np.mean(psnr_scores)}')
    print(f'SSIM: {np.mean(ssim_scores)}')
    print(f'FID Score: {fid_score}')
    print(f"Average KID: {np.mean(all_kid_values)}")
    # print(f'TSED: {np.mean(tsed_scores)}')

if __name__ == "__main__":
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/real-estate/rgb'
    gt_folder_type = 'fast-DiT/data/real-estate/rgb'
    # filename = 'frame_000440.jpg'
    generated_dir = os.path.join(base_dir, folder_type)
    ground_truth_dir = os.path.join(base_dir, gt_folder_type)
    ###########Json file for pose and intrinsics############
    json_file = 'fast-DiT/data/real-estate/output.json'
    with open(os.path.join(base_dir, json_file), 'r') as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: x['timestamp'])
    # Create a dictionary for quick lookup
    data_dict = {entry['timestamp']: entry for entry in data}
    # Iterate through the data to find the specific timestamp
    generated_img_files = sorted(os.listdir(generated_dir))
    # Initialize variables for pose and intrinsics
    n = len(generated_img_files)
    poses = [] #np.zeros((n, 4, 4))
    intrinsics = [] #np.zeros((n, 3, 3))
    W = 640    ##real estatte dataset size
    H = 360    ##real estatte dataset size
    for image_file in generated_img_files:     ###########generated imaGE IN 256
        # Extract timestamp from filename
        filename = os.path.basename(image_file)
        file_timestamp = int(os.path.splitext(filename)[0])  # Assuming filename is just the timestamp
        if file_timestamp in data_dict:
            entry = data_dict[file_timestamp]
            # for i, entry in enumerate(sorted_data):
            # Load pose
            pose = np.array(entry['pose'])
            poses.append(pose)
            # Load and transform intrinsics
            intrinsic = np.array(entry['intrinsics'])
            ############## uncomment following converting for real-estate dataset intrinsics
            intrinsic[0, :] = intrinsic[0, :] * W 
            intrinsic[1, :] = intrinsic[1, :] * H    
            
            scale1 = 256/min(H, W)  # final feature map size is 32
            scale2 = 256/(min(H, W) * (intrinsic[1, 1]/intrinsic[0, 0]))
            
            intrinsic[0, 0] = intrinsic[0, 0] * scale1
            intrinsic[1, 1] = intrinsic[1, 1] * scale2
            intrinsic[0, 2] = 128
            intrinsic[1, 2] = 128 
            ################ uncomment following converting for scannet and scannetv2 dataset intrinsics 
            # scale1 = 256/min(H, W) #########final feature map size is 32
            # scale2 = 256/(min(H, W) * (intrinsic[1, 1]/intrinsic[0, 0]))
            # intrinsic[0, 0] = intrinsic[0, 0] * scale1
            # intrinsic[1, 1] = intrinsic[1, 1] * scale2
            # intrinsic[0, 2] = 128 #954.7021
            # intrinsic[1, 2] = 128 #723.6698

            intrinsics.append(intrinsic)

    ############scale and crop the gt folder from the original size to 256, same size of our generated image size############
    gt_img_files = sorted(os.listdir(ground_truth_dir))
    output_folder_type = "fast-DiT/data/real-estate/rgb"
    for gt_image in gt_img_files:     ###########GT IN 256
        print("######gt image########")
        print(gt_image)
        gt_img = cv2.imread(os.path.join(base_dir, gt_folder_type, gt_image))
        gt_cropped_img = center_crop_img_and_resize(gt_img, 256)
        filename = os.path.basename(image_file)
        print(filename)
        ### file_timestamp = int(os.path.splitext(filename)[0])
        cv2.imwrite(os.path.join(base_dir, output_folder_type, filename), gt_cropped_img)
    cropped_ground_truth_dir = os.path.join(base_dir, output_folder_type)

    evaluate(generated_dir, cropped_ground_truth_dir)      ######First 5 metric results on poxels of generated vs GT images########
    tsed_evaluate(generated_dir, poses, intrinsics)     ##### Pixel consistency
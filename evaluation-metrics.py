import os
import numpy as np
import torch
import lpips
import cv2
from torchvision import models, transforms
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm

# Load the LPIPS model
lpips_model = lpips.LPIPS(net='alex')

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

def compute_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())

def compute_tsed(seq1, seq2):
    # Placeholder function for TSED. Implement TSED calculation logic here.
    return np.random.rand()

import cv2
import numpy as np
import os

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

def get_min_dist(p1, E, kp):
    epipolar_line = np.dot(p1, E.T)
    norm = np.linalg.norm(epipolar_line[:2])
    epipolar_line /= norm
    
    kp_h = np.append(kp, 1)
    s = -np.dot(epipolar_line, kp_h) / norm
    
    return np.abs(s)

def compute_tsed(img1, img2, pose1, pose2, intrinsics, threshold=1.0):
    points1, points2, matches = get_sift_matches(img1, img2)
    
    if len(matches) == 0:
        return 0
    
    E12 = get_essential_matrix(pose1, pose2, intrinsics)
    E21 = get_essential_matrix(pose2, pose1, intrinsics)
    
    seds = []
    for p1, p2 in zip(points1, points2):
        sed1 = get_min_dist(p1, E12, p2)
        sed2 = get_min_dist(p2, E21, p1)
        sed = 0.5 * (sed1 + sed2)
        seds.append(sed)
    
    n_matches = len(seds)
    median_sed = np.median(seds) if n_matches > 0 else 99999999999
    
    return median_sed

# Example usage in the evaluate function
def tsed_evaluate(generated_dir, ground_truth_dir, poses, intrinsics):
    generated_files = sorted(os.listdir(generated_dir))
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    
    tsed_scores = []
    
    for i in range(len(generated_files) - 1):
        gen_image1 = cv2.imread(os.path.join(generated_dir, generated_files[i]))
        gen_image2 = cv2.imread(os.path.join(generated_dir, generated_files[i + 1]))
        gt_image1 = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i]))
        gt_image2 = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i + 1]))
        
        pose1 = poses[i]
        pose2 = poses[i + 1]
        
        tsed_score_gen = compute_tsed(gen_image1, gen_image2, pose1, pose2, intrinsics)
        tsed_score_gt = compute_tsed(gt_image1, gt_image2, pose1, pose2, intrinsics)
        
        tsed_scores.append((tsed_score_gen, tsed_score_gt))
    
    avg_tsed_gen = np.mean([score[0] for score in tsed_scores])
    avg_tsed_gt = np.mean([score[1] for score in tsed_scores])
    
    print(f'Average TSED (Generated): {avg_tsed_gen}')
    print(f'Average TSED (Ground Truth): {avg_tsed_gt}')


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

    for gen_file, gt_file in tqdm(zip(generated_files, ground_truth_files), total=len(generated_files)):
        gen_image = load_image(os.path.join(generated_dir, gen_file))
        gt_image = load_image(os.path.join(ground_truth_dir, gt_file))
        
        # Compute features using InceptionV3
        with torch.no_grad():
            real_feature = inception_model(gt_image.unsqueeze(0)).flatten(1).cpu().numpy()
            gen_feature = inception_model(gen_image.unsqueeze(0)).flatten(1).cpu().numpy()
        
        real_features.append(real_feature)
        generated_features.append(gen_feature)
        
        # Compute metrics
        lpips_score = compute_lpips(gen_image, gt_image)
        psnr_score = compute_psnr(gen_image, gt_image)
        ssim_score = compute_ssim(gen_image, gt_image)
        # Compute TSED if necessary
        # tsed_score = compute_tsed(gen_image.numpy(), gt_image.numpy())
        
        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        # tsed_scores.append(tsed_score)
    
    # Compute FID and IS
    real_features = np.vstack(real_features)
    generated_features = np.vstack(generated_features)
    
    fid_score = compute_fid(real_features, generated_features)
    is_score = compute_is(generated_features)
    
    print(f'FID Score: {fid_score}')
    print(f'Inception Score: {is_score}')
    print(f'LPIPS: {np.mean(lpips_scores)}')
    print(f'PSNR: {np.mean(psnr_scores)}')
    print(f'SSIM: {np.mean(ssim_scores)}')
    print(f'TSED: {np.mean(tsed_scores)}')

if __name__ == "__main__":
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/real-estate/rgb'
    img_folder_type = 'rgb'
    filename = 'frame_000440.jpg'
    generated_dir = os.pat
    ground_truth_dir = "path/to/ground/truth/images"
    evaluate(generated_dir, ground_truth_dir)
    # tsed_evaluate(generated_dir, ground_truth_dir)

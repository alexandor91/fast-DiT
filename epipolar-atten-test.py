import cv2
import numpy as np
import os 
# from matplotlib import pyplot as plt 


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quaternion (np.ndarray): 1D array representing the quaternion in the order [qw, qx, qy, qz].

    Returns:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
    """
    qw, qx, qy, qz = quaternion

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
    rotation_matrix = np.array([[r11, r12, r13],
                                 [r21, r22, r23],
                                 [r31, r32, r33]])

    return rotation_matrix

def compute_skew_symmetric(v):
    """
    Compute the skew-symmetric matrix from a 3D vector.
    
    Args:
        v (np.ndarray): 3x1 vector.
        
    Returns:
        M (np.ndarray): 3x3 skew-symmetric matrix.
    """
    M = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return M

def compute_fundamental_matrix(K, K2, R, t):
    """
    Compute the fundamental matrix from intrinsic matrix and relative pose.
    
    Args:
        K (np.ndarray): 3x3 intrinsic matrix. source view 
        K2 (np.ndarray): 3x3 intrinsic matrix. target view 
        R (np.ndarray): 3x3 relative rotation matrix from target to source view.
        t (np.ndarray): 3x1 relative translation vector from target to source view.
        
    Returns:
        F (np.ndarray): 3x3 fundamental matrix.
    """
    # Compute the essential matrix
    E = np.dot(K2.T, np.dot(compute_skew_symmetric(t), np.dot(R, K)))
    
    # Enforce the rank-2 constraint on the essential matrix
    U, S, Vh = np.linalg.svd(E)
    S[2] = 0
    E = np.dot(U, np.dot(np.diag(S), Vh))
    
    # Compute the fundamental matrix from the essential matrix
    # F = np.dot(np.linalg.pinv(K), np.dot(E, np.linalg.pinv(K.T)))
    F = np.dot(np.linalg.pinv(K2.T), np.dot(E, np.linalg.pinv(K)))
    # F = E
    
    return F

def drawlines(img1, img2, epiline, pt1, pt2): 
    
    print(img1.shape)
    row, column = img1.shape[0:2] 
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
      
    # for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
    color = tuple(np.random.randint(0, 255, 
                                    3).tolist()) 
    epiline = epiline.reshape(-1)
    print('##############')
    print(epiline.shape)
    print(epiline)    
    print(pt1)    
    print(pt2)    
    
    print(pt1.shape)
    print(pt2.shape)

    # Compute the line endpoints based on the image dimensions
    # x1, y1 = 0, int(-c / b)
    # x2, y2 = width, int(-(c + a * width) / b)
    x0, y0 = [0, int(-epiline[2] / epiline[1])]
    x1, y1 = [column, int(-(epiline[2] + epiline[0] * column) / epiline[1])] 
    ##########epipolar line on source image################
    img2 = cv2.line(img2,  
                    (int(x0), int(y0)), (int(x1), int(y1)), color, 1) 
    # img2 = cv2.line(img2, (int(dst_pt[0]), int(dst_pt[1])), (int(epipolar_line[0]/epipolar_line[2]), int(epipolar_line[1]/epipolar_line[2])), color, thickness)

    thickness = 2
    img1 = cv2.circle(img1, 
                        (int(pt1[0]), int(pt1[1])), 5, color, thickness) 
    img2 = cv2.circle(img2,  
                        (int(pt2[0]), int(pt2[1])), 5, color, thickness) 
    return img1, img2 

def visualize_epipolar_line(source_img, target_img, intrinsic, R_src_to_target, t_src_to_target, src_pt, alpha=0.5, scale_factor=0.8):
    """
    Visualizes the epipolar line corresponding to a source image pixel in the target image.

    Args:
        source_img: Source view image (numpy array).
        target_img: Target view image (numpy array).
        intrinsic_src: Camera intrinsic matrix of the source view (3x3 numpy array).
        R_src_to_target: Rotation matrix from source to target view (3x3 numpy array).
        t_src_to_target: Translation vector from source to target view (3x1 numpy array).
        src_pt: Pixel position in the source image (2x1 numpy array).
        alpha: Transparency value for the epipolar line (0.0 to 1.0).
        scale_factor: Scale factor for darkening the background image (0.0 to 1.0).

    Returns:
        A numpy array representing the combined image with the visualized epipolar line.
    """

    # Calculate the fundamental matrix
    #   K_src_inv = np.linalg.inv(intrinsic_src)
    #   essential_matrix = R_src_to_target.dot(t_src_to_target.T)
    #   F = K_src_inv.T.dot(essential_matrix).dot(K_src_inv)
    print('$$$$$$$fundamental$$$$$$$')
    print(R_src_to_target.shape)
    F = compute_fundamental_matrix(intrinsic, R_src_to_target, t_src_to_target)

    print(F.shape)
    # Convert source pixel to homogeneous coordinates
    src_pt_homog = np.append(src_pt, [1])

    # Project source pixel to target image plane using epipolar constraint
    epipolar_line = F.dot(src_pt_homog)

    # Normalize the epipolar line
    # epipolar_line /= epipolar_line[2]
    epipolar_line /= np.linalg.norm(epipolar_line[:2])
    # Reshape epipolar line for line drawing
    line = epipolar_line[:2].reshape(1, -1)

    # Get image heights and widths
    h_target, w_target = target_img.shape[:2]

    # Clip the epipolar line to image boundaries
    line = np.clip(line, a_min =0, a_max = np.amax([h_target, w_target]))

    # Draw the epipolar line on a copy of the target image
    target_img_vis = target_img.copy()
    print(line.shape)
    # cv2.line(target_img_vis, tuple(line[0]), tuple(line.squeeze()), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # # Dim the background image
    # target_img_vis = target_img_vis.astype(np.float32) * scale_factor

    # # Create a mask for the epipolar line with transparency
    # mask = np.ones_like(target_img_vis) * alpha
    # cv2.line(mask, tuple(line[0]), tuple(line.squeeze()), (1, 1, 1), thickness=2, lineType=cv2.LINE_AA)

    # # Combine the target image and epipolar line with transparency
    # vis_img = cv2.addWeighted(target_img_vis, 1 - mask, mask, alpha, 0)

    return None  #vis_img.astype(np.uint8)


def draw_sift_features_and_epipolar_lines(img1, img2, F):
    """
    Visualizes SIFT features and their corresponding epipolar lines in another image view.

    Args:
        img1: First image (numpy array).anyanyde
        img2: Second image (numpy array).
        F: Fundamental matrix (numpy array of shape (3, 3)).
    """

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find matches between keypoints using FLANN matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    print("##########matches found##########")
    # print(matches)
    # Filter good matches using ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.56*n.distance:
            good_matches.append(m)

    # Draw matches on both images
    # img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

    # Get source and target image keypoint locations
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Draw epipolar lines for each source keypoint in the target image
    for i in range(len(good_matches)):
        if i == 16:          #######match pair 16 is correct################
            # Get source and target keypoints
            src_pt = src_pts[i][0]
            dst_pt = dst_pts[i][0]

            # Calculate epipolar line using fundamental matrix and source keypoint
            epipolar_line = calculate_epipolar_line(src_pt, F)

            # Normalize the epipolar line
            # epipolar_line /= epipolar_line[2]

            # epipolar_line /= np.linalg.norm(epipolar_line[:3])
            # Reshape epipolar line for line drawing
            line = epipolar_line[:3].reshape(1, -1)
            print('#######epipolar line#######')
            print(line.shape)
            # Get image heights and widths
            h_target, w_target = img2.shape[:2]

            # Clip the epipolar line to image boundaries
            # line = np.clip(line, a_min =0, a_max = np.amax([h_target, w_target]))
            # Draw the epipolar line on the target image
            color = (0, 255, 0)  # Green for epipolar lines
            thickness = 2
            # img2 = cv2.line(img2, (int(dst_pt[0]), int(dst_pt[1])), (int(epipolar_line[0]/epipolar_line[2]), int(epipolar_line[1]/epipolar_line[2])), color, thickness)
            src_img, tar_img = drawlines(img1, img2, line, src_pt, dst_pt)

    # Show resulting images
    # cv2.imwrite()
    # cv2.imshow('Matches with SIFT Features', img_matches)
    # cv2.imshow('Target Image with Epipolar Lines', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("epipolar_line_matches.jpg", src_img)
    cv2.imwrite("ref_images_feats.jpg", tar_img)


def calculate_epipolar_line(point, fundamental_matrix):
    """
    Calculates the epipolar line corresponding to a point in the target view given the fundamental matrix.

    Args:
        point_target: A numpy array of shape (2,) representing the point coordinates (u, v) in the target view.
        fundamental_matrix: A numpy array of shape (3, 3) representing the fundamental matrix.

    Returns:
        A numpy array of shape (3,) representing the epipolar line in homogeneous coordinates.
    """

    # Convert target point to homogeneous coordinates
    point_homog = np.append(point, [1])

    # Calculate the epipolar line using the fundamental matrix
    epipolar_line = fundamental_matrix.dot(point_homog)
    

    # Normalize the epipolar line
    epipolar_line /= epipolar_line[2]
    # epipolar_line /= np.linalg.norm(epipolar_line[:3])

    return epipolar_line

if __name__ == "__main__":
    # Example usage

    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data'
    source_img = cv2.imread(os.path.join(base_dir, folder_type, 'frame_000440.jpg'))
    target_img = cv2.imread(os.path.join(base_dir, folder_type, 'frame_000470.jpg'))

    scene_id= '0a5c013435'
    focal_length_x = 1432.3682   #####target view  1431.4313
    focal_length_y = 1432.3682 ######target view  1431.4313
    principal_point_x = 954.08276   #######target view 954.7021
    principal_point_y = 724.18256 ####target view 723.6698


    # Replace with your actual camera calibration matrices
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
    ##########
    src_quatern = np.array([0.837752, 0.490157, -0.150019, 0.188181])
    src_trans = np.array([0.158608, 1.22818, -1.60956])   ############source frame##########
    tar_quatern = np.array([0.804066, 0.472639, -0.25357, 0.256501])
    tar_trans = np.array([0.473911, 1.28311, -1.5215])       ############target frame######

    src_rot_mat = quaternion_to_rotation_matrix(src_quatern)
    tar_rot_mat = quaternion_to_rotation_matrix(tar_quatern)

    relative_rot_mat = np.dot(tar_rot_mat, src_rot_mat.T)
    # Replace with your relative transformation between views
    R_target_to_src = np.eye(3)  # Rotation matrix (assuming no rotation for simplicity)
    src_homo_mat = np.vstack((np.hstack((src_rot_mat, src_trans[:, None])), [0, 0, 0 ,1]))    
    tar_homo_mat = np.vstack((np.hstack((tar_rot_mat, tar_trans[:, None])), [0, 0, 0 ,1]))    


    # src_homo_mat = np.array([[0.8850673192958701, 0.1673973226830824, 0.43432008389729243, 0.3589177096482855],
    #         [-0.4608850948147595, 0.44571824240272334, 0.7674109756697478, 0.7506100032463232],
    #         [-0.0651215786061575, -0.8793820788529829, 0.4716415695861629, 1.8568835837010935],
    #         [0.0, 0.0, 0.0, 1.000000119209119]])
    src_homo_mat = np.array([[0.42891303, 0.40586197, 0.8070378, 1.4285464], ###########raw source pose
            [-0.06427293, -0.8774122, 0.4754123, 1.6330968],
            [0.9010566, -0.25578123, -0.35024756, -1.2047926],
            [0.0, 0.0, 0.0, 0.99999964]])

    # tar_homo_mat = np.array([[0.7373248387059498, 0.1747686589059863, 0.6525395399728156, 0.41220507023838177],
    #         [-0.654558791857726, 0.4236908657529971, 0.6261295452837368, 0.7302791578359176],
    #         [-0.16704700086583693, -0.8887864590670669, 0.4267942421598756, 1.8661927343815823],
    #         [0.0, 0.0, 0.0, 0.9999999999997726]])

    tar_homo_mat = np.array([[0.19473474, 0.39851177, 0.8962516, 1.4587839], ########raw target pose
            [-0.1672538, -0.8868709, 0.43068102, 1.642482],
            [0.966491, -0.23377006, -0.106051974, -1.1564484],
            [0.0, 0.0, 0.0, 0.9999995]])
    # relative_homo_mat = np.dot(tar_homo_mat.T, src_homo_mat)
    relative_homo_mat = np.dot(tar_homo_mat, src_homo_mat.T)

    # Example source pixel position
    u = 600
    v = 500
    src_pt = np.array([u, v])
    print("#######relative homo######")
    # print(relative_homo_mat[:3, 3])
    # t_rel = tar_trans - tar_rot_mat @ src_rot_mat.T @ src_trans

    F = compute_fundamental_matrix(src_intrinsic, tar_intrinsic, relative_homo_mat[:3, :3], [relative_homo_mat[0,3], relative_homo_mat[1,3], relative_homo_mat[2,3]])

    draw_sift_features_and_epipolar_lines(source_img, target_img, F)

    # Detect the SIFT key points and  
    # compute the descriptors for the  
    # two images 

    # sift = cv2.xfeatures2d.SIFT_create() 
    # keyPointsLeft, descriptorsLeft = sift.detectAndCompute(source_img, 
    #                                                     None) 
    
    # keyPointsRight, descriptorsRight = sift.detectAndCompute(target_img, 
    #                                                         None) 
    
    # # Create FLANN matcher object 
    # FLANN_INDEX_KDTREE = 0
    # indexParams = dict(algorithm=FLANN_INDEX_KDTREE, 
    #                 trees=5) 
    # searchParams = dict(checks=50) 
    # flann = cv2.FlannBasedMatcher(indexParams, 
    #                             searchParams) 
    
    # matches = flann.knnMatch(keyPointsLeft, keyPointsRight, k=2)
    
    # # Apply ratio test 
    # goodMatches = [] 
    # ptsLeft = [] 
    # ptsRight = [] 
    
    # for m, n in matches: 
        
    #     if m.distance < 0.8 * n.distance: 
            
    #         goodMatches.append([m]) 
    #         ptsLeft.append(keyPointsLeft[m.trainIdx].pt) 
    #         ptsRight.append(keyPointsRight[n.trainIdx].pt) 
    # # Visualize the epipolar line


    # vis_img = visualize_epipolar_line(source_img, target_img, intrinsic, relative_homo_mat[:3, :3], [relative_homo_mat[0, 3], relative_homo_mat[1, 3] , relative_homo_mat[2, 3]], src_pt)

    # # Display the visualized image
    # cv2.imshow("Epipolar Line Visualization", vis_img)

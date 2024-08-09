import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def cross_entropy_loss_per_pixel(gt_image, gen_image):
    """
    Calculate the cross-entropy loss per pixel between two images and visualize the loss using a magma colormap.

    Parameters:
    gt_image (numpy array): Ground truth image.
    gen_image (numpy array): Generated image.

    Returns:
    None
    """
    # Ensure the images are in float32 format
    gt_image = gt_image.astype(np.float32)
    gen_image = gen_image.astype(np.float32)
    
    # Normalize images to range [0, 1]
    gt_image /= 255.0
    gen_image /= 255.0

    # Clip the values to avoid log(0)
    epsilon = 1e-14  # Smaller epsilon to avoid zero
    gen_image = np.clip(gen_image, epsilon, 1.0 - epsilon)
    gt_image = np.clip(gt_image, epsilon, 1.0 - epsilon)
    
    # Calculate cross-entropy loss per pixel
    loss = np.abs(gen_image - gt_image)# - (gt_image * np.log(gen_image) + (1 - gt_image) * np.log(1 - gen_image))
    
    # Visualize the loss using a magma colormap
    plt.imshow(loss, cmap='magma')
    # Add a color bar
    plt.colorbar()  # This adds the color bar
    
    plt.axis('off')

    if not os.path.exists('./data/CrossEntropyResults'):
        os.makedirs('./data/CrossEntropyResults')
    plt.savefig('./data/CrossEntropyResults/cross_entropy_nvs11.png', dpi=600, bbox_inches='tight', pad_inches=0)

    # plt.show()

# Example usage
if __name__ == "__main__":
    base_dir = '/home/student.unimelb.edu.au/xueyangk'
    folder_type = 'fast-DiT/data/test-folder'
    img_folder_type = 'test-folder'
    # Load the images (replace with your image paths)
    gt_image_path = os.path.join(base_dir, folder_type, 'gt-0011.png')
    gen_image_path = os.path.join(base_dir, folder_type, 'nvs-0011.png')

    gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    gen_image = cv2.imread(gen_image_path, cv2.IMREAD_GRAYSCALE)
    # print(gt_image)
    # print(gen_image)
    if gt_image is None or gen_image is None:
        print("Error: One or both images could not be loaded.")
    else:
        cross_entropy_loss_per_pixel(gt_image, gen_image)
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Load the image
image_path = '/mnt/data/00000.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the image
normalized_image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Invert the normalized image
inverted_image = 1 - normalized_image

# Apply the 'magma' colormap
magma_colormap = cm.get_cmap('magma')
colored_image = magma_colormap(inverted_image)

# Convert the colormap image to 8-bit per channel
colored_image_8bit = (colored_image[:, :, :3] * 255).astype(np.uint8)

# Save the result
output_path = '/mnt/data/colored_image_flipped.jpg'
cv2.imwrite(output_path, cv2.cvtColor(colored_image_8bit, cv2.COLOR_RGB2BGR))

# Display the result
plt.imshow(colored_image_8bit)
plt.axis('off')
plt.show()
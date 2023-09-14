import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

import imageio
from scipy import ndimage
import cv2
import joblib

# Load the pre-trained RandomForestClassifier model
clf = joblib.load("image_segmentation/random_forest_model.pkl")

# Load the new image
img_new = imageio.imread("image_segmentation/image_0126.png")

# Perform image segmentation on the new image
sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)
features_new = features_func(img_new)
result_new = future.predict_segmenter(features_new, clf)

# Assuming you have a binary mask named 'label_mask' (True for the region of interest)
# Calculate the centroid of the mask for 'poles'
poles_mask = result_new == 1
centroid_poles = ndimage.measurements.center_of_mass(poles_mask)
center_x_poles, center_y_poles = centroid_poles

# Calculate the centroid of the mask for 'ground'
ground_mask = result_new == 2
centroid_ground = ndimage.measurements.center_of_mass(ground_mask)
center_x_ground, center_y_ground = centroid_ground

# Draw circles on the new image to mark the center points
cv2.circle(img_new, (int(center_y_poles), int(center_x_poles)), 10, (0, 255, 0), -1)
cv2.circle(img_new, (int(center_y_ground), int(center_x_ground)), 10, (0, 0, 255), -1)

# Plot the new image with center points and segmentation
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6, 4))
ax[0].imshow(segmentation.mark_boundaries(img_new, result_new, mode='thick'))
ax[0].set_title('Image')
ax[1].imshow(result_new)
ax[1].set_title('Segmentation')
ax[2].imshow(poles_mask)
ax[2].set_title('Poles Mask')
fig.tight_layout()

plt.show()

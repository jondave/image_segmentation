import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

import cv2
from scipy import ndimage
import joblib
import os
from PIL import Image

# Load the pre-trained RandomForestClassifier model
clf = joblib.load("random_forest_model.pkl")

# Load the MP4 video
video_path = "video_5fps.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a VideoWriter for the output video
output_video_path = 'output_video.mp4'
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Create a directory to save PNG frames
frame_rgb_output_dir = 'frames_rgb'
os.makedirs(frame_rgb_output_dir, exist_ok=True)
frame_result_output_dir = 'frames_result'
os.makedirs(frame_result_output_dir, exist_ok=True)
frame_mask_output_dir = 'frames_mask'
os.makedirs(frame_mask_output_dir, exist_ok=True)

frame_number = 0

while True:
    ret, frame = cap.read()

    # Break the loop when we reach the end of the video
    if not ret:
        break

    # Perform image segmentation on the frame
    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    features_frame = features_func(frame)
    result_frame = future.predict_segmenter(features_frame, clf)

    # Calculate the centroids and draw circles as before
    poles_mask = result_frame == 1
    centroid_poles = ndimage.measurements.center_of_mass(poles_mask)
    center_x_poles, center_y_poles = centroid_poles
    cv2.circle(frame, (int(center_y_poles), int(center_x_poles)), 10, (0, 255, 0), -1)

    ground_mask = result_frame == 2
    centroid_ground = ndimage.measurements.center_of_mass(ground_mask)
    center_x_ground, center_y_ground = centroid_ground
    cv2.circle(frame, (int(center_y_ground), int(center_x_ground)), 10, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame)

    # Save the frame as a PNG image
    frame_filename_rgb = os.path.join(frame_rgb_output_dir, f'frame_{frame_number:04d}.png')
    cv2.imwrite(frame_filename_rgb, frame)

    result_frame_image = Image.fromarray(result_frame)
    frame_filename_result = os.path.join(frame_result_output_dir, f'frame_{frame_number:04d}.png')
    result_frame_image.save(frame_filename_result)

    poles_mask_image = Image.fromarray(poles_mask)
    frame_filename_mask = os.path.join(frame_mask_output_dir, f'frame_{frame_number:04d}.png')
    poles_mask_image.save(frame_filename_mask)
    
    print("Frame: " + str(frame_number))
    frame_number += 1

# Release the video objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Video '{output_video_path}' has been created.")

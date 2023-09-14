import cv2
import os

# Directory containing your PNG images
image_directory = 'folder'

# Output video file name and parameters
output_video_file = 'video.mp4'
fps = 5  # Frames per second
frame_size = (640,  480)  # Frame size in pixels (width, height)

# List all PNG files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
image_files.sort()  # Ensure files are sorted in the correct order

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

# Loop through the image files and write them to the video
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

    # Resize the frame if necessary
    if frame.shape[:2] != frame_size:
        frame = cv2.resize(frame, frame_size)

    out.write(frame)

# Release the video writer
out.release()

print(f"Video '{output_video_file}' has been created.")

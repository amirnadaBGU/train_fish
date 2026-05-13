import os
from PIL import Image
import numpy as np

# Source and destination paths
dataset_path = "dataset"
output_path = "dataset2"

# Define the padding color as an RGB vector [R, G, B]
# Currently set to White: [255, 255, 255]
# Example for Red: [255, 0, 0], Example for Black: [0, 0, 0]
PADDING_COLOR = [255, 255, 255]

# Create the directory structure in the destination path
for subset in ["train", "vali", "test"]:
    os.makedirs(os.path.join(output_path, subset, "images"), exist_ok=True)

# Process each image
for subset in ["train", "vali", "test"]:
    images_dir = os.path.join(dataset_path, subset, "images")

    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)

        # Load the image and ensure it is in RGB mode
        img = Image.open(img_path)
        img = img.convert("RGB")
        img_array = np.array(img)

        # Change the padding to the specified color vector
        # 140 pixels from the top and bottom
        img_array[:140, :] = PADDING_COLOR  # Top part
        img_array[-140:, :] = PADDING_COLOR  # Bottom part

        # Save the new image
        new_img = Image.fromarray(img_array)
        output_img_path = os.path.join(output_path, subset, "images", img_name)
        new_img.save(output_img_path)

        print(f"Processing: {output_img_path}")

print("Finished!")
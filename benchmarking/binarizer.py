# Binarizes tiff stacks for inference evaluation

import os
import shutil
import numpy as np
import tifffile

# Define the directories
directories = [
    # "/another/dir"
    "/path/to/directory/with/your/tiff/stacks"
]

# Temporary directories to store binarized images
temp_dirs = [
    # "temp1
    "temp2"
]

# Ensure temporary directories exist
for temp_dir in temp_dirs:
    os.makedirs(temp_dir, exist_ok=True)

# Function to binarize a TIFF stack
def binarize_tiff(input_path, output_path):
    # Read the TIFF stack
    img = tifffile.imread(input_path)
    # Binarize the image: set all non-zero values to 255 (white)
    binarized_img = np.where(img > 0, 255, 0).astype(np.uint8)
    # Save the binarized image to the temporary directory
    tifffile.imwrite(output_path, binarized_img)

# Iterate through each directory and process the TIFF stacks
for dir_path, temp_dir in zip(directories, temp_dirs):
    for filename in os.listdir(dir_path):
        if filename.endswith(".tif"):
            input_path = os.path.join(dir_path, filename)
            temp_output_path = os.path.join(temp_dir, filename)
            # Binarize the TIFF stack and save to temporary directory
            binarize_tiff(input_path, temp_output_path)
            # Delete the original image
            os.remove(input_path)
            # Move the binarized image back to the original directory
            shutil.move(temp_output_path, input_path)

# Cleanup: Remove the temporary directories (optional)
for temp_dir in temp_dirs:
    shutil.rmtree(temp_dir)

print("Binarization and transfer complete.")

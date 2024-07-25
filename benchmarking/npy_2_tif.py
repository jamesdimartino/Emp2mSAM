# Converts list of .npy files to list of .tif files

import numpy as np
import tifffile
import os

# Define the paths
gt_paths = [
    "/path/to/your/first/groundtruth/segmentation.npy",
  "/path/to/your/second/groundtruth/segmentation.npy",
]

pred_paths = [
    "/path/to/your/first/prediction/segmentation.npy",
  "/path/to/your/second/prediction/segmentation.npy",

]
# Define the output directories
gt_output_dir = "/path/to/your/output/dir"
pred_output_dir = "/path/to/your/output/dir"

os.makedirs(gt_output_dir, exist_ok=True)
os.makedirs(pred_output_dir, exist_ok=True)

# Function to convert .npy to .tif
def convert_npy_to_tif(npy_path, tif_path):
    data = np.load(npy_path)
    tifffile.imwrite(tif_path, data)

# Convert gt_paths
# for gt_path in gt_paths:
   # filename = os.path.splitext(os.path.basename(gt_path))[0] + ".tif"
   # tif_path = os.path.join(gt_output_dir, filename)
   # convert_npy_to_tif(gt_path, tif_path)

# Convert pred_paths
for pred_path in pred_paths:
    filename = os.path.splitext(os.path.basename(pred_path))[0] + ".tif"
    tif_path = os.path.join(pred_output_dir, filename)
    convert_npy_to_tif(pred_path, tif_path)

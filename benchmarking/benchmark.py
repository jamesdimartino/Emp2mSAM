# Creates bounding boxes from ground truth segmentations and computes pixels accuracy per tile specifically for those bounding boxes, then averages across tiles.

import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from skimage.measure import regionprops, label

def load_tiff_stack(directory, filename):
    """Load a TIFF stack from a specified directory and filename."""
    return tiff.imread(os.path.join(directory, filename))

def calculate_pixel_accuracy_within_bbox(ground_truth, prediction, bbox):
    """Calculate the percentage of correctly labeled pixels within the bounding box."""
    min_row, min_col, max_row, max_col = bbox
    gt_bbox = ground_truth[min_row:max_row, min_col:max_col]
    pred_bbox = prediction[min_row:max_row, min_col:max_col]
    return np.sum(gt_bbox == pred_bbox), gt_bbox.size

def enlarge_bbox(bbox, factor, max_height, max_width):
    """Enlarge the bounding box by a given factor from its center."""
    min_row, min_col, max_row, max_col = bbox
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    height = max_row - min_row
    width = max_col - min_col

    new_height = height * factor
    new_width = width * factor

    new_min_row = max(0, int(center_row - new_height / 2))
    new_max_row = min(max_height, int(center_row + new_height / 2))
    new_min_col = max(0, int(center_col - new_width / 2))
    new_max_col = min(max_width, int(center_col + new_width / 2))

    return new_min_row, new_min_col, new_max_row, new_max_col

def main():
    ground_truth_dir = '/hpc/mydata/james.dimartino/Emp2mSAM/FILES/benching/groundtruth'
    prediction_dir = '/hpc/mydata/james.dimartino/Emp2mSAM/FILES/benching/mSAMveo'

    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    prediction_files = sorted(os.listdir(prediction_dir))

    assert len(ground_truth_files) == len(prediction_files), "Directories must contain the same number of files."

    enlargement_factor = 1.5
    tile_accuracies = []

    for gt_file, pred_file in tqdm(zip(ground_truth_files, prediction_files), total=len(ground_truth_files), desc="Processing stacks"):
        ground_truth_stack = load_tiff_stack(ground_truth_dir, gt_file)
        prediction_stack = load_tiff_stack(prediction_dir, pred_file)

        assert ground_truth_stack.shape == prediction_stack.shape, f"Shape mismatch: {gt_file} and {pred_file}"

        z_planes = ground_truth_stack.shape[0]
        stack_accuracies = []
        total_bboxes = 0

        for z in range(z_planes):
            gt_plane = ground_truth_stack[z]
            pred_plane = prediction_stack[z]
            gt_labeled = label(gt_plane)
            gt_props = regionprops(gt_labeled)

            correct_pixels = 0
            total_pixels = 0
            total_bboxes += len(gt_props)

            for prop in gt_props:
                bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
                enlarged_bbox = enlarge_bbox(bbox, enlargement_factor, gt_plane.shape[0], gt_plane.shape[1])
                correct, total = calculate_pixel_accuracy_within_bbox(gt_plane, pred_plane, enlarged_bbox)
                correct_pixels += correct
                total_pixels += total

            if total_pixels > 0:
                plane_accuracy = (correct_pixels / total_pixels) * 100
                stack_accuracies.append(plane_accuracy)

        if stack_accuracies:
            average_stack_accuracy = np.mean(stack_accuracies)
            tile_accuracies.append(average_stack_accuracy)
            print(f"Tile {gt_file}: Average Pixel Accuracy = {average_stack_accuracy:.2f}%, Total Bounding Boxes = {total_bboxes}")

    if tile_accuracies:
        overall_average_accuracy = np.mean(tile_accuracies)
        print(f"\nOverall Average Pixel Accuracy across all tiles = {overall_average_accuracy:.2f}%")

if __name__ == "__main__":
    main()

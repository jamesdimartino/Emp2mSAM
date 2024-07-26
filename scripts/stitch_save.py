import os
import shutil
import numpy as np
import tifffile
import psutil
import logging
import argparse
import cupy as cp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def print_memory_usage():
    """
    Logs the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    logging.info(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

def stitch_tiles_from_directory(input_dir, original_shape, tile_shape):
    """
    Stitch tiles from a directory into a single array.

    Args:
        input_dir (str): Directory containing input tile files.
        original_shape (tuple): Original shape of the full dataset.
        tile_shape (tuple): Shape of each tile.

    Returns:
        cp.ndarray: Stitched result array.
    """
    logging.info("Starting tile stitching...")
    print_memory_usage()

    stitched_result = cp.zeros(original_shape, dtype=cp.uint32)
    logging.info("Allocated memory for stitched result array.")
    print_memory_usage()

    tile_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    logging.info(f"Found {len(tile_files)} tile files in the directory.")

    max_label = 0
    for tile_file in tqdm(tile_files, desc="Stitching Tiles"):
        logging.info(f"Stitching tile {tile_file}")
        tile_path = os.path.join(input_dir, tile_file)
        temp_results = cp.load(tile_path, allow_pickle=True)

        # Ensure unique labels
        if temp_results.max() > 0:
            temp_results[temp_results > 0] += max_label
            max_label = temp_results.max().item()  # Convert to scalar

        # Extract index from file name
        try:
            idx = int(tile_file.split('_')[-1].split('.')[0])
        except ValueError:
            logging.error(f"Unable to parse index from filename: {tile_file}")
            continue

        row = (idx // 4) * tile_shape[1]
        col = (idx % 4) * tile_shape[2]
        logging.info(f"Row: {row}, Col: {col}")
        stitched_result[:, row:row+tile_shape[1], col:col+tile_shape[2]] = temp_results
        print_memory_usage()

    logging.info("Finished stitching all tiles.")
    print_memory_usage()

    return stitched_result

def update_label_mapping(label_mapping, label1, label2):
    """
    Update the label mapping to ensure transitive closure.

    Args:
        label_mapping (cp.ndarray): The current label mapping.
        label1 (int): First label to merge.
        label2 (int): Second label to merge.
    """
    # Find the current mappings for both labels
    root1 = label_mapping[label1]
    root2 = label_mapping[label2]
    
    # Determine the new root (minimum of the two)
    new_root = cp.minimum(root1, root2)
    
    # Update all labels pointing to root2 to point to root1
    label_mapping[label_mapping == root2] = new_root
    # Ensure transitive closure: update root1 to point to new_root
    label_mapping[root1] = new_root

def merge_across_stitches(stitched_result, tile_shape):
    """
    Merge labels across vertical and horizontal stitches using a mapping approach.

    Args:
        stitched_result (cp.ndarray): Stitched result array.
        tile_shape (tuple): Shape of each tile.
    """
    logging.info("Starting to merge instances across stitches...")

    depth, height, width = stitched_result.shape
    tile_height, tile_width = tile_shape[1], tile_shape[2]

    # Create a mapping array to store old to new label mappings
    max_label = int(stitched_result.max())
    label_mapping = cp.arange(max_label + 1, dtype=cp.uint32)

    # Vertical stitches
    for col in range(tile_width, width, tile_width):
        logging.info(f"Merging vertical stitch at column {col}")
        for z in tqdm(range(depth), desc=f"Vertical stitch at col {col}"):
            for row in range(height):
                left_pixel = stitched_result[z, row, col - 1]
                right_pixel = stitched_result[z, row, col]
                if left_pixel != right_pixel and left_pixel != 0 and right_pixel != 0:
                    update_label_mapping(label_mapping, left_pixel, right_pixel)

    # Horizontal stitches
    for row in range(tile_height, height, tile_height):
        logging.info(f"Merging horizontal stitch at row {row}")
        for z in tqdm(range(depth), desc=f"Horizontal stitch at row {row}"):
            for col in range(width):
                top_pixel = stitched_result[z, row - 1, col]
                bottom_pixel = stitched_result[z, row, col]
                if top_pixel != bottom_pixel and top_pixel != 0 and bottom_pixel != 0:
                    update_label_mapping(label_mapping, top_pixel, bottom_pixel)

    logging.info("Finished creating label mappings.")
    print_memory_usage()

    # Apply label mappings to the entire stack
    stitched_result = label_mapping[stitched_result]

    logging.info("Finished merging instances across stitches.")
    print_memory_usage()

    return stitched_result

def merge_and_save_slices_to_temp(stitched_result, temp_dir):
    """
    Save each slice directly to temporary files to avoid memory overflow.

    Args:
        stitched_result (cp.ndarray): Stitched result array.
        temp_dir (str): Directory to save temporary files.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    logging.info("Starting to save slices to temporary files...")
    
    depth = stitched_result.shape[0]

    for z in tqdm(range(depth), desc="Saving slices"):
        # Save each processed slice to a temporary file
        temp_file_path = os.path.join(temp_dir, f"slice_{z:04d}.npy")
        np.save(temp_file_path, cp.asnumpy(stitched_result[z]))

    logging.info("Finished saving slices to temporary files.")
    print_memory_usage()

def stack_temp_files_to_tiff(temp_dir, output_tiff_path, depth):
    """
    Stack temporary files into a single TIFF stack and delete the temporary files.

    Args:
        temp_dir (str): Directory containing temporary files.
        output_tiff_path (str): Path to save the output TIFF file.
        depth (int): Number of slices in the dataset.
    """
    logging.info("Starting to stack temporary files into TIFF...")

    with tifffile.TiffWriter(output_tiff_path, bigtiff=True) as tif:
        for z in tqdm(range(depth), desc="Stacking slices"):
            temp_file_path = os.path.join(temp_dir, f"slice_{z:04d}.npy")
            slice_data = np.load(temp_file_path)
            tif.write(slice_data, contiguous=True)

    logging.info("Finished stacking temporary files into TIFF.")
    print_memory_usage()

    # Delete the entire temporary directory
    shutil.rmtree(temp_dir)
    logging.info(f"Deleted temporary directory {temp_dir}")

def main(input_dir, output_tiff_path, original_shape, tile_shape):
    """
    Main function to stitch tiles and save the result as a TIFF file.

    Args:
        input_dir (str): Directory containing input tile files.
        output_tiff_path (str): Path to save the output TIFF file.
        original_shape (tuple): Original shape of the full dataset.
        tile_shape (tuple): Shape of each tile.
    """
    temp_dir = "temp_slices"

    # Stitch the 2D stacks together from the directory
    stitched_result = stitch_tiles_from_directory(input_dir, original_shape, tile_shape)
    logging.info(f"Stitched result shape: {stitched_result.shape}")

    # Merge instances across stitches
    stitched_result = merge_across_stitches(stitched_result, tile_shape)

    # Save slices to temporary files
    merge_and_save_slices_to_temp(stitched_result, temp_dir)

    # Stack temporary files into a single TIFF stack
    stack_temp_files_to_tiff(temp_dir, output_tiff_path, original_shape[0])

    logging.info(f"Stitched segmentation saved as {output_tiff_path}.")

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Stitch tiles and save as TIFF.')
    parser.add_argument('input_dir', type=str, help='Directory containing input tiles')
    parser.add_argument('output_tiff_path', type=str, help='Path to save the output TIFF file')
    parser.add_argument('--original_shape', type=int, nargs=3, required=True, help='Original shape of the full dataset (depth, height, width)')
    parser.add_argument('--tile_shape', type=int, nargs=3, required=True, help='Shape of each tile (depth, height, width)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_tiff_path, tuple(args.original_shape), tuple(args.tile_shape))

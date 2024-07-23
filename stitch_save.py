import os
import numpy as np
import tifffile
import psutil
import logging
import argparse
from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d

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
        np.ndarray: Stitched result array.
    """
    logging.info("Starting tile stitching...")
    print_memory_usage()

    stitched_result = np.zeros(original_shape, dtype=np.uint32)
    logging.info("Allocated memory for stitched result array.")
    print_memory_usage()

    tile_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    logging.info(f"Found {len(tile_files)} tile files in the directory.")

    for idx, tile_file in enumerate(tile_files):
        logging.info(f"Stitching tile {idx}: {tile_file}")
        tile_path = os.path.join(input_dir, tile_file)
        temp_results = np.load(tile_path, allow_pickle=True)

        row = (idx // 4) * tile_shape[1]
        col = (idx % 4) * tile_shape[2]
        logging.info(f"Row: {row}, Col: {col}")
        stitched_result[:, row:row+tile_shape[1], col:col+tile_shape[2]] = temp_results
        print_memory_usage()

    logging.info("Finished stitching all tiles.")
    print_memory_usage()

    return stitched_result

def main(input_dir, output_tiff_path, original_shape, tile_shape):
    """
    Main function to stitch tiles and save the result as a TIFF file.

    Args:
        input_dir (str): Directory containing input tile files.
        output_tiff_path (str): Path to save the output TIFF file.
        original_shape (tuple): Original shape of the full dataset.
        tile_shape (tuple): Shape of each tile.
    """
    # Stitch the 2D stacks together from the directory
    stitched_result = stitch_tiles_from_directory(input_dir, original_shape, tile_shape)
    logging.info(f"Stitched result shape: {stitched_result.shape}")

    # Save the stitched result as a TIFF file before attempting 3D merge
    stitched_output_tiff_path = output_tiff_path.replace('.tif', '_stitched.tif')
    tifffile.imwrite(stitched_output_tiff_path, stitched_result.astype(np.uint32))
    logging.info(f"Stitched segmentation saved as {stitched_output_tiff_path}.")

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


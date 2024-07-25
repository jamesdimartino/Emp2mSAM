import os
import numpy as np
import torch
from skimage.measure import regionprops
from segment_anything import SamPredictor
import micro_sam.util as util
from micro_sam.inference import batched_inference
import psutil
import logging
import argparse

logging.basicConfig(level=logging.INFO)

def print_memory_usage():
    """
    Print the current memory usage.
    """
    process = psutil.Process(os.getpid())
    logging.info(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

def load_tile(tile_path):
    """
    Load a tile from a file.

    Parameters:
    tile_path (str): Path to the tile file.

    Returns:
    numpy.ndarray: Loaded tile data.
    """
    logging.info(f"Loading tile from {tile_path}...")
    tile = np.load(tile_path, allow_pickle=True)
    logging.info(f"Tile loaded with shape: {tile.shape}")
    return tile

def extract_bounding_boxes(empanada_results, enlargement_factor):
    """
    Extract bounding boxes from empanada results.

    Parameters:
    empanada_results (list of numpy.ndarray): Empanada result tiles.
    enlargement_factor (float): Factor to enlarge bounding boxes.

    Returns:
    list: Extracted bounding boxes for each slice.
    """
    logging.info("Extracting bounding boxes from empanada results...")
    all_boxes = []
    for chunk_idx, chunk in enumerate(empanada_results):
        chunk_boxes = []
        for z_idx, z_slice in enumerate(chunk):
            props = regionprops(z_slice)
            slice_boxes = []
            for prop in props:
                bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
                min_col, min_row, max_col, max_row = bbox[1], bbox[0], bbox[3], bbox[2]
                if enlargement_factor != 1.0:
                    width = max_col - min_col
                    height = max_row - min_row
                    center_col = min_col + width / 2
                    center_row = min_row + height / 2

                    new_width = width * enlargement_factor
                    new_height = height * enlargement_factor

                    new_min_col = center_col - new_width / 2
                    new_max_col = center_col + new_width / 2
                    new_min_row = center_row - new_height / 2
                    new_max_row = center_row + new_height / 2

                    slice_boxes.append([new_min_col, new_min_row, new_max_col, new_max_row])
                else:
                    slice_boxes.append([min_col, min_row, max_col, max_row])
            chunk_boxes.append(slice_boxes)
        all_boxes.append(chunk_boxes)
    logging.info(f"Total bounding boxes extracted for all chunks: {sum(len(chunk) for chunk in all_boxes)}")
    return all_boxes

def run_sam_inference(predictor, image, boxes, batch_size=1):
    """
    Run SAM inference on an image with given bounding boxes.

    Parameters:
    predictor (SamPredictor): SAM model predictor.
    image (numpy.ndarray): Image data.
    boxes (list): Bounding boxes for the image.
    batch_size (int): Batch size for inference.

    Returns:
    numpy.ndarray: SAM segmentation mask.
    """
    logging.info(f"Running SAM inference on image of shape {image.shape} with batch size {batch_size}...")
    boxes = np.array(boxes)
    logging.info(f"Boxes shape: {boxes.shape}")
    seg_mask = batched_inference(predictor, image, batch_size, boxes=boxes)
    logging.info("SAM inference completed.")
    return seg_mask

def process_and_save_tile(predictor, raw_tile, boxes, tile_idx, output_dir):
    """
    Process and save a tile with SAM inference.

    Parameters:
    predictor (SamPredictor): SAM model predictor.
    raw_tile (numpy.ndarray): Raw tile data.
    boxes (list): Bounding boxes for the tile.
    tile_idx (int): Index of the tile.
    output_dir (str): Directory to save the SAM results.
    """
    z_sam_results = np.zeros((raw_tile.shape[0], raw_tile.shape[1], raw_tile.shape[2]), dtype=np.uint32)
    for z_idx in range(raw_tile.shape[0]):
        if not boxes[tile_idx][z_idx]:
            logging.info(f"Skipping SAM inference for tile {tile_idx}, Z-plane {z_idx} due to no bounding boxes")
            continue

        image = raw_tile[z_idx, :, :]
        logging.info(f"Running SAM inference for tile {tile_idx}, Z-plane {z_idx}")
        sam_result = run_sam_inference(predictor, image, boxes[tile_idx][z_idx])
        z_sam_results[z_idx, :, :] = sam_result

    logging.info(f"Stacked SAM results shape for tile {tile_idx}: {z_sam_results.shape}")

    output_path = os.path.join(output_dir, f"sam_result_{tile_idx:04d}.npy")
    np.save(output_path, z_sam_results)
    logging.info(f"SAM results saved to {output_path}")
    del z_sam_results  # Clear memory

def load_empanada_results(empanada_dir):
    """
    Load empanada results from a directory.

    Parameters:
    empanada_dir (str): Directory containing empanada result files.

    Returns:
    list: Loaded empanada result tiles.
    """
    logging.info(f"Loading empanada results from directory: {empanada_dir}...")
    empanada_files = sorted([os.path.join(empanada_dir, f) for f in os.listdir(empanada_dir) if f.endswith('.npy')])
    empanada_results = [np.load(file, allow_pickle=True) for file in empanada_files]
    logging.info(f"Loaded {len(empanada_results)} empanada result files.")
    return empanada_results

def main(input_dir, empanada_dir, output_dir, enlargement_factor, tile_shape):
    """
    Main function to run SAM inference on all tiles.

    Parameters:
    input_dir (str): Directory containing input tiles.
    empanada_dir (str): Directory containing empanada result files.
    output_dir (str): Directory to save SAM results.
    enlargement_factor (float): Factor to enlarge bounding boxes.
    tile_shape (tuple): Shape of the tiles (height, width, depth).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    tile_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    empanada_results = load_empanada_results(empanada_dir)

    boxes = extract_bounding_boxes(empanada_results, enlargement_factor)

    predictor = util.get_sam_model("vit_l_em_organelles", device=device)

    for tile_idx, tile_file in enumerate(tile_files):
        tile_path = os.path.join(input_dir, tile_file)
        raw_tile = load_tile(tile_path)
        process_and_save_tile(predictor, raw_tile, boxes, tile_idx, output_dir)
        print_memory_usage()

    logging.info("Batched inference completed and results saved.")

def parse_args():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run SAM inference on all tiles.')
    parser.add_argument('input_dir', type=str, help='Directory containing input tiles')
    parser.add_argument('empanada_dir', type=str, help='Directory containing empanada result files')
    parser.add_argument('output_dir', type=str, help='Directory to save SAM results')
    parser.add_argument('--enlargement_factor', type=float, default=1, help='Factor to enlarge bounding boxes')
    parser.add_argument('--tile_shape', type=int, nargs=3, default=(500, 1024, 1024), help='Shape of the tiles (depth, height, width)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.input_dir, args.empanada_dir, args.output_dir, args.enlargement_factor, tuple(args.tile_shape))


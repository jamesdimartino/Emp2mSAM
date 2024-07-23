import os
import numpy as np
import torch
from skimage.measure import regionprops
from segment_anything import SamPredictor
import micro_sam.util as util
from micro_sam.inference import batched_inference
import psutil
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)

def print_memory_usage():
    """
    Logs the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    logging.info(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

def load_tile(tile_path):
    """
    Loads a tile from a specified path.

    Args:
        tile_path (str): Path to the tile file.

    Returns:
        np.ndarray: Loaded tile.
    """
    logging.info(f"Loading tile from {tile_path}...")
    tile = np.load(tile_path, allow_pickle=True)
    logging.info(f"Tile loaded with shape: {tile.shape}")
    return tile

def load_sam_results(sam_dir):
    """
    Loads SAM results from a specified directory.

    Args:
        sam_dir (str): Directory containing SAM result files.

    Returns:
        list of np.ndarray: List of loaded SAM result files.
    """
    logging.info(f"Loading SAM results from directory: {sam_dir}...")
    sam_files = sorted([os.path.join(sam_dir, f) for f in os.listdir(sam_dir) if f.endswith('.npy')])
    sam_results = [np.load(file, allow_pickle=True) for file in sam_files]
    logging.info(f"Loaded {len(sam_results)} SAM result files.")
    return sam_results

def extract_centroids_and_points(sam_results, min_area=2000):
    """
    Extracts centroids and points from SAM results for small masks and removes small masks.

    Args:
        sam_results (list of np.ndarray): List of SAM result files.
        min_area (int): Minimum area threshold for masks to be retained.

    Returns:
        tuple:
            - all_points (list): List of points for small masks.
            - all_point_labels (list): List of point labels for small masks.
            - cleaned_sam_results (list of np.ndarray): List of cleaned SAM results with small masks removed.
    """
    logging.info("Extracting centroids and points from SAM results...")
    all_points = []
    all_point_labels = []
    cleaned_sam_results = []
    for tile_idx, tile in enumerate(tqdm(sam_results, desc="Processing tiles")):
        tile_points = []
        tile_point_labels = []
        cleaned_tile = []
        for z_idx, z_slice in enumerate(tile):
            props = regionprops(z_slice)
            slice_points = []
            slice_point_labels = []
            cleaned_slice = np.zeros_like(z_slice)
            instance_number = 1
            for prop in props:
                if prop.area < min_area:
                    centroid = prop.centroid
                    slice_points.append([[centroid[1], centroid[0]]])  # X, Y coordinates
                    slice_point_labels.append([1])  # Positive prompt
                else:
                    cleaned_slice[z_slice == prop.label] = instance_number
                    instance_number += 1
            tile_points.append(slice_points)
            tile_point_labels.append(slice_point_labels)
            cleaned_tile.append(cleaned_slice)
        all_points.append(tile_points)
        all_point_labels.append(tile_point_labels)
        cleaned_sam_results.append(np.array(cleaned_tile))
    logging.info("Centroids and points extraction completed.")
    return all_points, all_point_labels, cleaned_sam_results

def run_sam_inference(predictor, image, points, point_labels, batch_size=1):
    """
    Runs SAM inference on an image using specified points and point labels.

    Args:
        predictor (SamPredictor): The SAM predictor object.
        image (np.ndarray): The input image.
        points (np.ndarray): The point prompts.
        point_labels (np.ndarray): The point labels.
        batch_size (int): The batch size for inference.

    Returns:
        np.ndarray: The segmentation mask.
    """
    logging.info(f"Running SAM inference on image of shape {image.shape} with batch size {batch_size}...")
    points = np.array(points)
    point_labels = np.array(point_labels)
    logging.info(f"Points shape: {points.shape}, Point labels shape: {point_labels.shape}")
    seg_mask = batched_inference(predictor, image, batch_size, points=points, point_labels=point_labels)
    logging.info("SAM inference completed.")
    return seg_mask

def process_and_save_tile(predictor, raw_tile, points, point_labels, cleaned_sam_results, tile_idx, output_dir):
    """
    Processes and saves a tile after running SAM inference and combining results.

    Args:
        predictor (SamPredictor): The SAM predictor object.
        raw_tile (np.ndarray): The raw tile image.
        points (list): List of points for SAM inference.
        point_labels (list): List of point labels for SAM inference.
        cleaned_sam_results (list of np.ndarray): List of cleaned SAM results.
        tile_idx (int): Index of the current tile.
        output_dir (str): Directory to save the output results.
    """
    concatenated_results = np.zeros((raw_tile.shape[0], raw_tile.shape[1], raw_tile.shape[2]), dtype=np.uint32)
    max_uint32 = np.iinfo(np.uint32).max
    for z_idx in range(raw_tile.shape[0]):
        if not points[tile_idx][z_idx]:
            logging.info(f"Skipping SAM inference for tile {tile_idx}, Z-plane {z_idx} due to no points")
            concatenated_results[z_idx, :, :] = cleaned_sam_results[tile_idx][z_idx]
            continue

        image = raw_tile[z_idx, :, :]
        logging.info(f"Running SAM inference for tile {tile_idx}, Z-plane {z_idx}")
        sam_result = run_sam_inference(predictor, image, points[tile_idx][z_idx], point_labels[tile_idx][z_idx])

        max_label = cleaned_sam_results[tile_idx][z_idx].max()
        sam_result[sam_result > 0] += max_label

        concatenated_result = cleaned_sam_results[tile_idx][z_idx] + sam_result

        # Ensure no pixel values exceed the maximum value for np.uint32
        concatenated_result = np.clip(concatenated_result, 0, max_uint32)

        concatenated_results[z_idx, :, :] = concatenated_result

    logging.info(f"Stacked SAM results shape for tile {tile_idx}: {concatenated_results.shape}")

    output_path = os.path.join(output_dir, f"sam_result_{tile_idx:04d}.npy")
    np.save(output_path, concatenated_results)
    logging.info(f"SAM results saved to {output_path}")
    del concatenated_results  # Clear memory

def main(sam_dir, raw_dir, output_dir):
    """
    Main function to run the SAM inference pipeline.

    Args:
        sam_dir (str): Directory containing SAM1 results.
        raw_dir (str): Directory containing raw tile images.
        output_dir (str): Directory to save SAM2 results.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    sam_results = load_sam_results(sam_dir)
    points, point_labels, cleaned_sam_results = extract_centroids_and_points(sam_results, min_area=2000)

    predictor = util.get_sam_model("vit_l_em_organelles", device=device)

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.npy')])
    for tile_idx, raw_file in enumerate(tqdm(raw_files, desc="Processing raw files")):
        raw_path = os.path.join(raw_dir, raw_file)
        raw_tile = load_tile(raw_path)
        process_and_save_tile(predictor, raw_tile, points, point_labels, cleaned_sam_results, tile_idx, output_dir)
        print_memory_usage()

    logging.info("Second iteration of batched inference completed and results saved.")

def parse_args():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the SAM inference pipeline.')
    parser.add_argument('sam_dir', type=str, help='Directory containing SAM1 results')
    parser.add_argument('raw_dir', type=str, help='Directory containing raw tile images')
    parser.add_argument('output_dir', type=str, help='Directory to save SAM2 results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.sam_dir, args.raw_dir, args.output_dir)


import os
import numpy as np
import cupy as cp
import argparse
from skimage.measure import regionprops
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import cdist
from tqdm import tqdm
import networkx as nx
import gc

def get_args():
    parser = argparse.ArgumentParser(description="Merge 2D instance segmentations into 3D using an object tracking-based approach.")
    parser.add_argument('input_dir', type=str, help='Directory containing input .npy files')
    parser.add_argument('output_dir', type=str, help='Directory to save output .npy files')
    parser.add_argument('--distance_threshold', type=float, default=20, help='Distance threshold for centroid matching')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of tiles to process in a batch')
    return parser.parse_args()

def load_tile(file_path):
    print(f"Loading tile from {file_path}")
    return np.load(file_path)

def save_tile(file_path, data):
    print(f"Saving tile to {file_path}")
    np.save(file_path, data)
    print(f"Tile saved successfully to {file_path}")

def clear_gpu_memory():
    print("Clearing GPU memory")
    cp._default_memory_pool.free_all_blocks()
    gc.collect()

def calculate_centroids_and_pixel_coords(labels):
    props = regionprops(labels)
    centroids = [prop.centroid for prop in props]
    pixel_coords = [prop.coords for prop in props]
    return centroids, pixel_coords

def track_instances(tile, distance_threshold):
    z_slices = tile.shape[0]
    centroids_by_slice = []
    pixel_coords_by_slice = []

    for z in range(z_slices):
        current_labels, num_features = nd_label(cp.asnumpy(tile[z]))
        centroids, pixel_coords = calculate_centroids_and_pixel_coords(current_labels)
        centroids_by_slice.append(centroids)
        pixel_coords_by_slice.append(pixel_coords)

    G = nx.DiGraph()

    for z in tqdm(range(z_slices - 2), desc="Tracking instances"):
        centroids_current = centroids_by_slice[z]
        centroids_next = centroids_by_slice[z + 1]
        centroids_next_next = centroids_by_slice[z + 2]

        for i, centroid_current in enumerate(centroids_current):
            for j, centroid_next in enumerate(centroids_next):
                dist = cdist([centroid_current], [centroid_next], 'euclidean')[0][0]
                if dist < distance_threshold:
                    # Check if the instances in the next slice merge into one instance in the subsequent slice
                    merged = False
                    for k, centroid_next_next in enumerate(centroids_next_next):
                        dist_next = cdist([centroid_next], [centroid_next_next], 'euclidean')[0][0]
                        if dist_next < distance_threshold:
                            merged = True
                            break
                    if merged:
                        G.add_edge((z, i), (z + 1, j), weight=1.0 / (1.0 + dist))

    # Convert the graph to undirected to find connected components
    paths = list(nx.connected_components(G.to_undirected()))

    new_labels = np.zeros_like(tile)
    label_counter = 1

    print("Assigning labels to connected components...")
    for path in tqdm(paths, desc="Assigning labels"):
        for (z, idx) in path:
            for coord in pixel_coords_by_slice[z][idx]:
                new_labels[z][coord[0], coord[1]] = label_counter
        label_counter += 1

    print(f"Total number of unique objects: {label_counter - 1}")

    return new_labels
def process_tile(tile_file, input_dir, output_dir, distance_threshold):
    print(f"Processing tile: {tile_file}")
    tile = cp.array(load_tile(os.path.join(input_dir, tile_file)))

    new_labels = track_instances(cp.asnumpy(tile), distance_threshold)

    # Save the labeled image only
    save_tile(os.path.join(output_dir, f"merged_{tile_file}"), new_labels)
    clear_gpu_memory()

def merge_2d_to_3d(input_dir, output_dir, distance_threshold, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} files to process")

    for batch_start in tqdm(range(0, len(files), batch_size), desc="Processing batches"):
        batch_files = files[batch_start:batch_start + batch_size]

        for tile_file in batch_files:
            process_tile(tile_file, input_dir, output_dir, distance_threshold)

if __name__ == "__main__":
    args = get_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    merge_2d_to_3d(args.input_dir, args.output_dir, args.distance_threshold, args.batch_size)
    print("Processing complete.")

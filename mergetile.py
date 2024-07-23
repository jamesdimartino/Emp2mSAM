import os
import numpy as np
import psutil
import nifty
import elf.tracking.tracking_utils as track_utils
import elf.segmentation as seg_utils
from skimage.measure import regionprops, label
from skimage.segmentation import relabel_sequential
from scipy.ndimage import binary_closing
import logging
import argparse
import gc  # Import garbage collection module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to print memory usage
def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    logging.info(f"[{stage}] Memory usage: {mem:.2f} GB")

# Preprocessing function to perform gap closing
def _preprocess_closing(slice_segmentation, gap_closing):
    print_memory_usage("Start gap closing")
    binarized = slice_segmentation > 0
    structuring_element = np.zeros((3, 1, 1))
    structuring_element[:, 0, 0] = 1
    closed_segmentation = binary_closing(binarized, iterations=gap_closing, structure=structuring_element)
    print_memory_usage("After binary closing")

    new_segmentation = np.zeros_like(slice_segmentation)
    n_slices = new_segmentation.shape[0]

    offset = 1
    for z in range(n_slices):
        seg_z = slice_segmentation[z]
        seg_new = np.zeros_like(seg_z)  # Initialize seg_new to ensure it is always defined

        if z < gap_closing or z >= (n_slices - gap_closing):
            seg_z, _, _ = relabel_sequential(seg_z, offset=offset)
            offset = int(seg_z.max()) + 1
        else:
            closed_z = label(closed_segmentation[z])
            matches = nifty.ground_truth.overlap(closed_z, seg_z)
            matches = {seg_id: matches.overlapArrays(seg_id, sorted=False)[0]
                       for seg_id in range(1, int(closed_z.max() + 1))}
            matches = {k: v[v != 0] for k, v in matches.items()}

            ids_initial, ids_closed = [], []
            for seg_id, matched in matches.items():
                if len(matched) > 1:
                    ids_initial.extend(matched.tolist())
                else:
                    ids_closed.append(seg_id)

            closed_mask = np.isin(closed_z, ids_closed)
            seg_new[closed_mask] = closed_z[closed_mask]

            if ids_initial:
                initial_mask = np.isin(seg_z, ids_initial)
                seg_new[initial_mask] = relabel_sequential(seg_z[initial_mask], offset=seg_new.max() + 1)[0]

            seg_new, _, _ = relabel_sequential(seg_new, offset=offset)
            max_z = seg_new.max()
            if max_z > 0:
                offset = int(max_z) + 1

        new_segmentation[z] = seg_new
        # print_memory_usage(f"Processed slice {z}/{n_slices}")

    print_memory_usage("End gap closing")
    return new_segmentation

def process_tile(tile_path, output_dir, gap_closing=3, min_z_extent=5, beta=0.5):
    # Load the tile into memory
    tile = np.load(tile_path)
    logging.info(f"Loaded tile from {tile_path} with shape: {tile.shape}")
    print_memory_usage("After loading tile")

    # Apply the preprocessing gap closing step
    closed_tile = _preprocess_closing(tile, gap_closing)
    logging.info("Gap closing completed")
    print_memory_usage("After gap closing")

    # Compute edges from overlap
    edges = track_utils.compute_edges_from_overlap(closed_tile, verbose=False)
    logging.info(f"Number of edges: {len(edges)}")
    if len(edges) > 0:
        logging.info(f"Sample edges: {edges[:5]}")
    else:
        logging.error("No edges found, something might be wrong with the segmentation input.")
    print_memory_usage("After edge computation")

    # Prepare uv_ids and overlaps
    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])
    logging.info("Prepared uv_ids and overlaps")
    print_memory_usage("After preparing uv_ids and overlaps")

    # Create graph and insert edges
    n_nodes = int(closed_tile.max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    logging.info("Graph created successfully")
    print_memory_usage("After creating graph")

    valid_edges = []
    for i, edge in enumerate(uv_ids):
        if edge[0] != edge[1] and 0 <= edge[0] < n_nodes and 0 <= edge[1] < n_nodes:
            graph.insertEdge(edge[0], edge[1])
            valid_edges.append(edge)
    logging.info(f"Edges inserted successfully, total valid edges: {len(valid_edges)}")
    print_memory_usage("After inserting edges")

    valid_uv_ids = np.array(valid_edges)
    valid_overlaps = np.array([overlaps[i] for i, edge in enumerate(uv_ids) if list(edge) in valid_uv_ids.tolist()])
    logging.info("Valid uv_ids and overlaps calculated")
    print_memory_usage("After calculating valid overlaps")

    # Compute edge costs
    costs = seg_utils.multicut.compute_edge_costs(valid_overlaps)
    logging.info("Edge costs computed successfully")
    print_memory_usage("After computing edge costs")

    # Set background costs
    with_background = True
    if with_background:
        for i, edge in enumerate(valid_uv_ids):
            if edge[0] == 0 or edge[1] == 0:
                costs[i] = -8.0
        logging.info("Background costs set successfully")
        print_memory_usage("After setting background costs")

    # Perform multicut decomposition
    node_labels = seg_utils.multicut.multicut_decomposition(graph, 1.0 - costs, beta=beta)
    logging.info("Multicut decomposition completed successfully")
    print_memory_usage("After multicut decomposition")

    # Apply node labels to the segmentation
    logging.info(f"closed_tile shape: {closed_tile.shape}")
    logging.info(f"node_labels shape: {node_labels.shape}")
    segmentation = nifty.tools.take(node_labels, closed_tile)
    logging.info("Node labels applied successfully")
    print_memory_usage("After applying node labels")

    # Filter small segments if min_z_extent is specified
    if min_z_extent is not None and min_z_extent > 0:
        props = regionprops(segmentation)
        filter_ids = [prop.label for prop in props if (prop.bbox[3] - prop.bbox[0]) < min_z_extent]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        logging.info("Filtered small segments successfully")
        print_memory_usage("After filtering small segments")

    # Save the merged tile
    output_path = os.path.join(output_dir, os.path.basename(tile_path))
    np.save(output_path, segmentation)
    logging.info(f"Stored merged tile to {output_path}")
    print_memory_usage("Saved merged tile")

def main():
    parser = argparse.ArgumentParser(description="Process a single tile and merge segmentations.")
    parser.add_argument("tile_path", type=str, help="Path to the tile file")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed tile")
    args = parser.parse_args()

    process_tile(args.tile_path, args.output_dir)

if __name__ == "__main__":
    main()


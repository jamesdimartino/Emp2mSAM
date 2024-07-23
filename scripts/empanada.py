import os
import numpy as np
from skimage import io
import psutil
import time
import argparse

def run_empanada_inference(tile, config_path, temp_tile_path, tile_idx, args):
    """
    Run empanada inference on a tile and save the result.

    Parameters:
    tile (numpy.ndarray): Input tile data.
    config_path (str): Path to the empanada config file.
    temp_tile_path (str): Path to save the temporary tile file.
    tile_idx (int): Index of the current tile.
    args (argparse.Namespace): Parsed command line arguments.
    
    Returns:
    numpy.ndarray: Result of the empanada inference.
    """
    print(f"Running empanada inference on tile {tile_idx} of shape {tile.shape}...")

    # Save the tile as a temporary TIFF file
    io.imsave(temp_tile_path, tile)

    # Build the command with optional arguments
    command = f"python empanada_inference.py {config_path} {temp_tile_path}"
    if args.data_key:
        command += f" -data-key {args.data_key}"
    if args.mode:
        command += f" -mode {args.mode}"
    if args.qlen:
        command += f" -qlen {args.qlen}"
    if args.label_divisor:
        command += f" -nmax {args.label_divisor}"
    if args.seg_thr:
        command += f" -seg-thr {args.seg_thr}"
    if args.nms_thr:
        command += f" -nms-thr {args.nms_thr}"
    if args.nms_kernel:
        command += f" -nms-kernel {args.nms_kernel}"
    if args.iou_thr:
        command += f" -iou-thr {args.iou_thr}"
    if args.ioa_thr:
        command += f" -ioa-thr {args.ioa_thr}"
    if args.pixel_vote_thr:
        command += f" -pixel-vote-thr {args.pixel_vote_thr}"
    if args.cluster_iou_thr:
        command += f" -cluster-iou-thr {args.cluster_iou_thr}"
    if args.min_size:
        command += f" -min-size {args.min_size}"
    if args.min_span:
        command += f" -min-span {args.min_span}"
    if args.downsample_f:
        command += f" -downsample-f {args.downsample_f}"
    if args.one_view:
        command += " --one-view"
    if args.fine_boundaries:
        command += " --fine-boundaries"
    if args.use_cpu:
        command += " --use-cpu"
    if args.save_panoptic:
        command += " --save-panoptic"
    print(f"Executing command: {command}")
    os.system(command)

    result_path = temp_tile_path.replace(".tif", "_mito.tif")
    result = io.imread(result_path)
    # Delete temporary files
    os.remove(temp_tile_path)
    os.remove(result_path)

    print(f"Empanada inference completed for tile {tile_idx} with result shape {result.shape}")
    return result

def log_memory_usage(stage):
    """
    Log the current memory usage.

    Parameters:
    stage (str): Stage of the processing.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{stage}] Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

def process_tiles(input_dir, config_path, output_dir, args):
    """
    Process all tiles in the input directory and run empanada inference on each.

    Parameters:
    input_dir (str): Directory containing input tiles.
    config_path (str): Path to the empanada config file.
    output_dir (str): Directory to save empanada results.
    args (argparse.Namespace): Parsed command line arguments.
    """
    tile_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

    if not tile_files:
        raise FileNotFoundError(f"No .npy files found in directory: {input_dir}")

    for tile_idx, tile_file in enumerate(tile_files):
        print(f"Processing tile {tile_idx}: {tile_file}")
        tile_path = os.path.join(input_dir, tile_file)
        tile = np.load(tile_path, allow_pickle=True)
        temp_tile_path = f"temp_tile_{tile_idx}.tif"

        try:
            log_memory_usage(f"Before inference for tile {tile_idx}")
            start_time = time.time()
            empanada_result = run_empanada_inference(tile, config_path, temp_tile_path, tile_idx, args)
            end_time = time.time()
            log_memory_usage(f"After inference for tile {tile_idx}")

            # Save the result
            output_path = os.path.join(output_dir, f"empanada_result_{tile_idx:04d}.npy")
            np.save(output_path, empanada_result)
            print(f"Empanada result saved to {output_path}")
            print(f"Time taken for tile {tile_idx}: {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"Error processing tile {tile_idx}: {e}")
            continue

    print(f"Empanada results saved to {output_dir}")

def parse_args():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Runs empanada model inference.')
    parser.add_argument('input_dir', type=str, metavar='input_dir', help='Directory containing input tiles')
    parser.add_argument('output_dir', type=str, metavar='output_dir', help='Directory to save empanada results')
    parser.add_argument('-data-key', type=str, metavar='data-key', default=None,
                        help='Key in zarr volume (if volume_path is a zarr). For multiple keys, separate with a comma.')
    parser.add_argument('-mode', type=str, dest='mode', metavar='inference_mode', choices=['orthoplane', 'stack'],
                        default=None, help='Pick orthoplane (xy, xz, yz) or stack (xy)')
    parser.add_argument('-qlen', type=int, dest='qlen', metavar='qlen', choices=[1, 3, 5, 7, 9, 11],
                        default=None, help='Length of median filtering queue, an odd integer')
    parser.add_argument('-nmax', type=int, dest='label_divisor', metavar='label_divisor',
                        default=None, help='Maximum number of objects per instance class allowed in volume.')
    parser.add_argument('-seg-thr', type=float, dest='seg_thr', metavar='seg_thr', default=None,
                        help='Segmentation confidence threshold (0-1)')
    parser.add_argument('-nms-thr', type=float, dest='nms_thr', metavar='nms_thr', default=None,
                        help='Centroid confidence threshold (0-1)')
    parser.add_argument('-nms-kernel', type=int, dest='nms_kernel', metavar='nms_kernel', default=None,
                        help='Minimum allowed distance, in pixels, between object centers')
    parser.add_argument('-iou-thr', type=float, dest='iou_thr', metavar='iou_thr', default=None,
                        help='Minimum IoU score between objects in adjacent slices for label stiching')
    parser.add_argument('-ioa-thr', type=float, dest='ioa_thr', metavar='ioa_thr', default=None,
                        help='Minimum IoA score between objects in adjacent slices for label merging')
    parser.add_argument('-pixel-vote-thr', type=int, dest='pixel_vote_thr', metavar='pixel_vote_thr', default=None,
                        choices=[1, 2, 3], help='Votes necessary per voxel when using orthoplane inference')
    parser.add_argument('-cluster-iou-thr', type=float, dest='cluster_iou_thr', metavar='cluster_iou_thr', default=None,
                        help='Minimum IoU to group together instances after orthoplane inference')
    parser.add_argument('-min-size', type=int, dest='min_size', metavar='min_size', default=None,
                        help='Minimum object size, in voxels, in the final 3d segmentation')
    parser.add_argument('-min-span', type=int, dest='min_span', metavar='min_span', default=None,
                        help='Minimum number of consecutive slices that object must appear on in final 3d segmentation')
    parser.add_argument('-downsample-f', type=int, dest='downsample_f', metavar='dowsample_f', default=None,
                        help='Factor by which to downsample images before inference, must be log base 2.')
    parser.add_argument('--one-view', action='store_true', help='One to allow instances seen in just 1 stack through to orthoplane consensus.')
    parser.add_argument('--fine-boundaries', action='store_true', help='Whether to calculate cells on full resolution image.')
    parser.add_argument('--use-cpu', action='store_true', help='Whether to force inference to run on CPU.')
    parser.add_argument('--save-panoptic', action='store_true', help='Whether to save raw panoptic segmentation for each stack.')
    return parser.parse_args()

def main():
    """
    Main function to run empanada inference on all tiles.
    """
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    config_path = "/hpc/mydata/james.dimartino/Emp2mSAM/empavars/configs/inference.yaml"  # Hardcoded path to empanada config

    os.makedirs(output_dir, exist_ok=True)
    process_tiles(input_dir, config_path, output_dir, args)

if __name__ == "__main__":
    main()


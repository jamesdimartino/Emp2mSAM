import os
import argparse

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the entire EM data processing pipeline.')
    parser.add_argument('input_tiff', type=str, help='Path to the input TIFF file')
    parser.add_argument('output_dir', type=str, help='Directory to save intermediate and final results')
    parser.add_argument('--block_size', type=int, default=500, help='Number of slices per block')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of slices per chunk')
    parser.add_argument('--tile_shape', type=int, nargs=2, default=(1024, 1024), help='Shape of each tile (height, width)')
    parser.add_argument('--original_shape', type=int, nargs=3, required=True, help='Original shape of the full dataset (depth, height, width)')
    return parser.parse_args()


def run_tiler(input_tiff, output_dir, block_size, chunk_size, tile_shape):
    os.system(f"python tiler.py {input_tiff} {output_dir}/tiles --block_size {block_size} --chunk_size {chunk_size} --tile_shape {tile_shape[0]} {tile_shape[1]}")

def run_empanada(input_dir, output_dir):
    os.system(f"python empanada.py {input_dir} {output_dir}")

def run_sam(input_dir, empanada_dir, output_dir):
    os.system(f"python samboxes.py {input_dir} {empanada_dir} {output_dir}")

def run_points(sam_dir, raw_dir, output_dir):
    os.system(f"python sampoints.py {sam_dir} {raw_dir} {output_dir}")

def run_merge(input_dir, output_dir):
    os.system(f"python mergealltiles.py {input_dir} {output_dir}")

def run_stitch(input_dir, output_tiff_path, original_shape, tile_shape):
    os.system(f"python stitch_save.py {input_dir} {output_tiff_path} --original_shape {original_shape[0]} {original_shape[1]} {original_shape[2]} --tile_shape {tile_shape[0]} {tile_shape[1]} {tile_shape[2]}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    tiles_dir = os.path.join(args.output_dir, "tiles")
    empanada_results_dir = os.path.join(args.output_dir, "empanada_results")
    sam_results_dir = os.path.join(args.output_dir, "sam_results")
    sam2_results_dir = os.path.join(args.output_dir, "sam2_results")
    merged_dir = os.path.join(args.output_dir, "merged")
    
    run_tiler(args.input_tiff, args.output_dir, args.block_size, args.chunk_size, args.tile_shape)
    run_empanada(tiles_dir, empanada_results_dir)
    run_sam(tiles_dir, empanada_results_dir, sam_results_dir)
    run_points(sam_results_dir, tiles_dir, sam2_results_dir)
    run_merge(sam2_results_dir, merged_dir)
    run_stitch(merged_dir, os.path.join(args.output_dir, "final_result.tif"), args.original_shape, args.tile_shape)

if __name__ == "__main__":
    main()


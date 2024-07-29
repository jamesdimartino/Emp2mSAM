import os
import argparse
import numpy as np
import torch
import micro_sam.util as util
import tifffile
import zarr

def convert_tiff_to_npy(tiff_path):
    print(f"Loading TIFF stack from {tiff_path}.")
    tiff_stack = tifffile.imread(tiff_path)
    print(f"TIFF stack loaded. Shape: {tiff_stack.shape}")
    return tiff_stack

def initialize_zarr_path(save_path):
    if os.path.exists(save_path):
        print(f"Save path {save_path} already exists. Checking for existing datasets.")
        zarr_store = zarr.open(save_path, mode='a')
        if 'features' in zarr_store:
            print(f"Removing existing 'features' dataset in {save_path}.")
            del zarr_store['features']
    else:
        print(f"Save path {save_path} does not exist. It will be created.")
    return save_path

def main(image_volume_path, save_path, tile_shape, halo):
    print("Initializing SAM model predictor with GPU support.")
    predictor = util.get_sam_model("vit_l", device='cuda')
    print("Predictor initialized.")

    print(f"Loading image volume from {image_volume_path}.")
    image_volume = convert_tiff_to_npy(image_volume_path)
    print(f"Image volume loaded. Shape: {image_volume.shape}")

    save_path = initialize_zarr_path(save_path)

    print("Preparing to call precompute_image_embeddings.")
    print(f"Save path: {save_path}")
    print(f"Tile shape: {tile_shape}")
    print(f"Halo: {halo}")
    print(f"Verbose: True")

    try:
        print("Starting embeddings computation.")
        embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_volume,
            save_path=save_path,
            tile_shape=tile_shape,
            halo=halo,
            verbose=True
        )
        print("Embeddings computation finished successfully.")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory. Attempting to release memory and retry.")
            torch.cuda.empty_cache()
            print("Memory cleared. Retrying embeddings computation.")
            embeddings = util.precompute_image_embeddings(
                predictor=predictor,
                input_=image_volume,
                save_path=save_path,
                tile_shape=tile_shape,
                halo=halo,
                verbose=True
            )
            print("Embeddings computation finished successfully after retry.")
        else:
            print(f"RuntimeError encountered: {e}")
            raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image embeddings using SAM predictor with GPU support.")
    parser.add_argument("image_volume_path", type=str, help="Path to the image volume file (TIFF stack).")
    parser.add_argument("save_path", type=str, help="Path to save the embeddings (zarr format).")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=None, help="Shape of tiles for tiled prediction.")
    parser.add_argument("--halo", type=int, nargs=2, default=None, help="Overlap of the tiles for tiled prediction.")

    args = parser.parse_args()

    print("Parsed arguments:")
    print(f"  image_volume_path: {args.image_volume_path}")
    print(f"  save_path: {args.save_path}")
    print(f"  tile_shape: {args.tile_shape}")
    print(f"  halo: {args.halo}")

    print("Starting main function.")
    main(args.image_volume_path, args.save_path, args.tile_shape, args.halo)
    print("Main function finished.")

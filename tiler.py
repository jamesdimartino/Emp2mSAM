import numpy as np
import tifffile
import torch
import psutil
import os
import argparse

def load_tiff_stack(file_path, start_slice, end_slice):
    """
    Load a chunk of a TIFF stack.

    Parameters:
    file_path (str): Path to the TIFF file.
    start_slice (int): Starting slice index.
    end_slice (int): Ending slice index.

    Returns:
    numpy.ndarray: Loaded chunk of the TIFF stack.
    """
    print(f"Loading TIFF stack from slice {start_slice} to {end_slice}...")
    stack = tifffile.imread(file_path, key=range(start_slice, end_slice))
    print(f"TIFF stack chunk loaded with shape {stack.shape}")
    return stack

def tile_chunk(chunk, tile_shape):
    """
    Tile a chunk of the TIFF stack into smaller tiles.

    Parameters:
    chunk (numpy.ndarray): Chunk of the TIFF stack.
    tile_shape (tuple): Shape of the tiles (depth, height, width).

    Returns:
    numpy.ndarray: Array of tiled chunks.
    """
    print(f"Tiling chunk of shape {chunk.shape} into tiles of shape {tile_shape}...")
    depth, height, width = chunk.shape
    tiles = []
    for i in range(0, height, tile_shape[1]):
        for j in range(0, width, tile_shape[2]):
            tile = chunk[:, i:i+tile_shape[1], j:j+tile_shape[2]]
            # If the tile is smaller than the desired shape, pad it
            if tile.shape[1] != tile_shape[1] or tile.shape[2] != tile_shape[2]:
                padded_tile = np.zeros((depth, tile_shape[1], tile_shape[2]), dtype=chunk.dtype)
                padded_tile[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded_tile
            tiles.append(tile)
    print(f"Chunk tiled into {len(tiles)} tiles.")
    return np.array(tiles)

def save_tile(tile, index, directory):
    """
    Save a tile to a file.

    Parameters:
    tile (numpy.ndarray): Tile to save.
    index (int): Index of the tile.
    directory (str): Directory to save the tile.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, f"tile_{index:04d}.npy"), tile)

def print_memory_usage(stage):
    """
    Print the current memory usage.

    Parameters:
    stage (str): Stage of the processing.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"[{stage}] Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def main(tiff_path, output_dir, block_size, chunk_size, tile_shape):
    """
    Main function to process the TIFF stack.

    Parameters:
    tiff_path (str): Path to the TIFF file.
    output_dir (str): Directory to save the tiles.
    block_size (int): Number of slices per block.
    chunk_size (int): Number of slices per chunk.
    tile_shape (tuple): Shape of the tiles (height, width).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get the total number of slices in the TIFF stack
    with tifffile.TiffFile(tiff_path) as tif:
        n_slices = len(tif.pages)

    # Initialize the tile index
    tile_index = 0

    # Process the TIFF stack in blocks
    for start_slice in range(0, n_slices, block_size):
        end_slice = min(start_slice + block_size, n_slices)
        print(f"Processing block from slice {start_slice} to {end_slice}")

        all_block_tiles = []

        # Process each block in chunks
        for chunk_start in range(start_slice, end_slice, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_slice)
            print(f"Processing chunk from slice {chunk_start} to {chunk_end}")
            chunk = load_tiff_stack(tiff_path, chunk_start, chunk_end)
            print_memory_usage("After loading chunk")

            # Transfer to GPU if available
            chunk = torch.tensor(chunk, device=device)
            print_memory_usage("After transferring to GPU")

            # Create tiles from the chunk
            chunk_tiles = tile_chunk(chunk.cpu().numpy(), (chunk_end - chunk_start, *tile_shape))
            print_memory_usage("After tiling chunk")

            all_block_tiles.append(chunk_tiles)

        # Stack the chunked tiles to form final tiles of shape (500, 1024, 1024)
        print(f"Stacking tiles for block from slice {start_slice} to {end_slice}")
        all_block_tiles = np.vstack(all_block_tiles)
        print_memory_usage("After vstack all_block_tiles")

        for i in range(16):
            print(f"Stacking tile {i+1} of 16")
            stacked_tile = np.concatenate(all_block_tiles[i::16], axis=0)
            print_memory_usage(f"After stacking tile {i+1} of 16")

            # Save the tile to the specified directory
            save_tile(stacked_tile, tile_index, output_dir)
            tile_index += 1
            del stacked_tile
            torch.cuda.empty_cache()
            print_memory_usage(f"After saving and clearing memory for tile {i+1} of 16")

        # Clear the memory of the processed block
        del all_block_tiles
        torch.cuda.empty_cache()
        print_memory_usage("After clearing memory of all_block_tiles")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TIFF stack into tiles.")
    parser.add_argument("tiff_path", type=str, help="Path to the TIFF file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the tiles.")
    parser.add_argument("--block_size", type=int, default=500, help="Number of slices per block.")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of slices per chunk.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(1024, 1024), help="Shape of the tiles (height, width).")

    args = parser.parse_args()
    main(args.tiff_path, args.output_dir, args.block_size, args.chunk_size, tuple(args.tile_shape))


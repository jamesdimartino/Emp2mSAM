# EM Data Processing Pipeline Documentation

---

## Emp2mSAM Overview Documentation

---

## Background Information

This pipeline processes electron microscopy (EM) data by following a series of steps that include tiling the data, running inference with the Empanada model, performing SAM inference, extracting points, merging tiles, and finally stitching the results into a single TIFF file. This end-to-end pipeline ensures efficient processing of large EM datasets while maintaining the integrity and quality of the data.

The pipeline is designed to handle large datasets by breaking them into manageable tiles and chunks, performing processing on each piece, and then reassembling the processed data. The use of GPU resources where available, along with memory management techniques, ensures that the pipeline can handle large volumes of data efficiently.

---

## Prerequisites

Ensure you have the following prerequisites installed in a clean environment:

```bash
# Install CUDA 12.1
pip install cuda=12.1

# Install cuDNN
pip install cudnn

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Empanada
conda install empanada-dl

# Install Micro-SAM
mamba install -c pytorch -c conda-forge micro_sam
```

---

## Tiler Script Documentation (tiler.py)

This script processes a TIFF stack into smaller tiles for further analysis.

### Functions

- `load_tiff_stack(file_path, start_slice, end_slice)`: Loads a chunk of a TIFF stack.
- `tile_chunk(chunk, tile_shape)`: Tiles a chunk of the TIFF stack into smaller tiles.
- `save_tile(tile, index, directory)`: Saves a tile to a file.
- `print_memory_usage(stage)`: Prints the current memory usage.
- `main(tiff_path, output_dir, block_size, chunk_size, tile_shape)`: Main function to process the TIFF stack.

### Usage


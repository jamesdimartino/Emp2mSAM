# Emp2mSAM Pipeline User Guide

---

## Background Information

This pipeline processes electron microscopy (EM) data to obtain mitochondrial segmentations by following a series of steps:
1. If larger than (:, 1024, 1024), the data is tiled into smaller chunks to be processed.
2. Initial segmentations for the tiled data are obtained with the Empanada model.
3. The empanada segmentations are converted to box prompts, which are used by the microSAM model to generate secondary segmentations.
4. The segmentations obtained from the microSAM model are filtered for small segmentations. Centroids are obtained for segmentations below the threshold, which are used as point prompts for the microSAM model to generate tertiary segmentations. These are combined with the secondary segmentations into a final segmentation set.
5. The final segmentations are merged in 3D space to obtain 3D instance segmentations of the mitochondria
6. The tiled results are stitched into a single tiff stack the size of the original image volume.

The pipeline can be run end-to-end, with the only inputs given being the input pathway to a TIFF volume, the desired output location, the shape of the volume, the desired shape of tiles for processing, and the block and chunk sizes. It can also be run in individual poritions by calling the scripts below one at a time.

The pipeline is designed to handle large datasets by breaking them into manageable tiles and chunks, performing processing on each piece, and then reassembling the processed data. The use of GPU resources where available, along with memory management techniques, ensures that the pipeline can handle large volumes of data efficiently.

---

## Prerequisites

Install the following prerequisites in the following order in a clean environment USING PYTHON 3.9:

```bash
# Install CUDA 12.1
conda install cuda=12.1

# Install cuDNN
conda install cudnn

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Empanada
conda install empanada-dl

# Install Micro-SAM
mamba install -c pytorch -c conda-forge micro_sam

# Install cupy-cuda12x
pip install cupy-cuda12x

# Install distinctipy
pip install distinctipy
```

Then, navigate to the directory you wish to operate from and run:

```bash
git clone https://github.com/jamesdimartino/Emp2mSAM.git
cd Emp2mSAM
cd scripts
```

Now, you can run the pipeline end-to-end or stepwise:


---

## run_end2end.py

This script runs the entire EM data processing pipeline end-to-end.

### Functions

- `parse_args()`: Parses command line arguments.
- `run_tiler(input_tiff, output_dir, block_size, chunk_size, tile_shape)`: Runs the Tiler script.
- `run_empanada(input_dir, output_dir)`: Runs the Empanada inference script.
- `run_sam(input_dir, empanada_dir, output_dir)`: Runs the SAM inference script.
- `run_points(sam_dir, raw_dir, output_dir)`: Runs the SAM points extraction script.
- `run_merge(input_dir, output_dir)`: Runs the tile merging script.
- `run_stitch(input_dir, output_tiff_path, original_shape, tile_shape)`: Runs the tile stitching script.
- `main()`: Main function to run the entire EM data processing pipeline.

### Arguments

- `tiff_path`: File path to raw tiff volume.
- `output_dir`: File path for results.
- `chunk_size`: Number of z slices from the original volume to process at a time. May require some trial and error to balance speed with memory consumption, default is 100.
- `block_size`: Number of z slices per chunk to process at a time. May require some trial and error to balance speed with memory consumption, default is 500.
- `original_shape`: Original shape of the input volume.
- `tile_shape`: Desired shape of tiles for data processing. Ideally, this should be 1024x1024, as this is the best input image shape for SAM.

### Usage

```bash
python pipeline.py <tiff_path> <output_dir> --block_size <block_size> --chunk_size <chunk_size> --tile_shape <height> <width> --original_shape <depth> <height> <width>
```

---

## tiler.py

This script processes a TIFF stack into smaller tiles. The TIFF stack is loaded in chunks of z planes, which are then tiled into blocks. These blocks are then stacked to create tiles that span the full z length of the stack. For example, starting with an original image with dimensions (500, 4096, 4096), tile_shape of (1024, 1024), chunk_size of 100, and block_size of 500, the data would first be divided into 5 (100, 4096, 4096) chunks, which would then be further divided into 80 (100, 1024, 1024) blocks, before being stacked into 16 (500, 1024, 1024) tiles, the shape used for further processing.

### Functions

- `load_tiff_stack(file_path, start_slice, end_slice)`: Loads a chunk of a TIFF stack.
- `tile_chunk(chunk, tile_shape)`: Tiles a chunk of the TIFF stack into smaller tiles.
- `save_tile(tile, index, directory)`: Saves a tile to a file.
- `print_memory_usage(stage)`: Prints the current memory usage.
- `main(tiff_path, output_dir, block_size, chunk_size, tile_shape)`: Main function to process the TIFF stack.

### Arguments

- `tiff_path`: File path to raw tiff volume.
- `output_dir`: File path for results.
- `chunk_size`: Number of z slices from the original volume to process at a time. May require some trial and error to balance speed with memory consumption, default is 100.
- `block_size`: Number of z slices per chunk to process at a time. May require some trial and error to balance speed with memory consumption, default is 500.
- `tile_shape`: Desired shape of tiles for data processing. Ideally, this should be 1024x1024, as this is the best input image shape for SAM.


### Usage
```bash
python tiler.py <tiff_path> <output_dir> --block_size <block_size> --chunk_size <chunk_size> --tile_shape <height> <width>
```


---

## empanada.py

This script runs the Empanada model inference on the tiles generated by the Tiler script.

### Functions

- `run_empanada_inference(tile, config_path, temp_tile_path, tile_idx, args)`: Runs Empanada inference on a tile and saves the result.
- `log_memory_usage(stage)`: Logs the current memory usage.
- `process_tiles(input_dir, config_path, output_dir, args)`: Processes all tiles in the input directory and runs Empanada inference on each.
- `parse_args()`: Parses command line arguments.
- `main()`: Main function to run Empanada inference on all tiles.

### Arguments

- `input_dir`: File path to input directory.
- `output_dir`: File path for results directory.


### Usage

```bash
python empanada.py <input_dir> <output_dir>
```


---

## samboxes.py

This script runs SAM inference on the tiles using bounding boxes extracted from Empanada results.

### Functions

- `print_memory_usage()`: Logs the current memory usage.
- `load_tile(tile_path)`: Loads a tile from a specified path.
- `extract_bounding_boxes(empanada_results, enlargement_factor)`: Extracts bounding boxes from Empanada results.
- `run_sam_inference(predictor, image, boxes, batch_size)`: Runs SAM inference on an image with given bounding boxes.
- `process_and_save_tile(predictor, raw_tile, boxes, tile_idx, output_dir)`: Processes and saves a tile with SAM inference.
- `load_empanada_results(empanada_dir)`: Loads Empanada results from a directory.
- `main(input_dir, empanada_dir, output_dir, enlargement_factor, tile_shape)`: Main function to run SAM inference on all tiles.
- `parse_args()`: Parses command line arguments.

### Arguments

- `input_dir`: File path to raw image tiles.
- `empanada_dir`: File path to directory where empanada results are stored.
- `output_dir`: File path for results.
- `enlargement_factor`: NOT RECOMMENDED. Factor by which to enlarge bounding boxes before passing them to microSAM. Default is 1.
- `tile_shape`: Height and width of tiles being passed to the script.

### Usage

```bash
python samboxes.py <input_dir> <empanada_dir> <output_dir> --enlargement_factor <factor> --tile_shape <height> <width>
```


---

## sampoints.py

This script filters small masks, obtains centroids from them, and runs SAM inference again.

### Functions

- `print_memory_usage()`: Logs the current memory usage.
- `load_tile(tile_path)`: Loads a tile from a specified path.
- `load_sam_results(sam_dir)`: Loads SAM results from a specified directory.
- `extract_centroids_and_points(sam_results, min_area)`: Extracts centroids and points from SAM results for small masks and removes small masks.
- `run_sam_inference(predictor, image, points, point_labels, batch_size)`: Runs SAM inference on an image using specified points and point labels.
- `process_and_save_tile(predictor, raw_tile, points, point_labels, cleaned_sam_results, tile_idx, output_dir)`: Processes and saves a tile after running SAM inference and combining results.
- `main(sam_dir, raw_dir, output_dir)`: Main function to run the SAM inference pipeline.
- `parse_args()`: Parses command line arguments.

### Arguments

- `sam_dir`: File path to directory where microSAM box outputs are stored.
- `raw_dir`: File path to directory where raw image tiles are stored.
- `output_dir`: File path for results directory.

### Usage

```bash
python sampoints.py <sam_dir> <raw_dir> <output_dir>
```

---

## mergealltiles.py

This script merges tiles using a multicut segmentation approach.

### Functions

- `print_memory_usage(stage)`: Logs the current memory usage.
- `process_tile(tile_path, output_dir, gap_closing, min_z_extent, beta)`: Processes a single tile and merges segmentations.
- `parse_args()`: Parses command line arguments.
- `main()`: Main function to merge tiles using the mergetile.py script.

### Arguments

- `tile_path`: File path to directory where segmentation tiles are stored. This is probably SAM points results but may be empanada results, SAM boxes results, or SAM points results depending on what you want to merge.
- `output_dir`: File path for results directory.

### Usage

```bash
python mergealltiles.py <tile_path> <output_dir>
```

---

## stitch_save.py

This script stitches tiles from a directory into a single array and saves it as a TIFF file. Additionally, it merges across stitch lines to create continuous images in 3D space.

### Functions

- `print_memory_usage()`: Logs the current memory usage.
- `stitch_tiles_from_directory(input_dir, original_shape, tile_shape)`: Stitches tiles from a directory into a single array.
- `main(input_dir, output_tiff_path, original_shape, tile_shape)`: Main function to stitch tiles and save the result as a TIFF file.
- `parse_args()`: Parses command line arguments.

### Arguments

- `tile_path`: File path to directory where segmentation tiles are stored. This is probably SAM points results but may be empanada results, SAM boxes results, or SAM points results depending on what you want to merge.
- `output_dir`: File path for results directory.
- `original_shape`: The original shape of the raw volume you are performing segmentation on, before tiling.
- `tile_shape`: The shape of the tiles being passed to the function.

### Usage

```bash
python stitch_save.py <input_dir> <output_tiff_path> --original_shape <depth> <height> <width> --tile_shape <depth> <height> <width>
```

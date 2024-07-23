import os
import subprocess
import argparse

def parse_args():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Merge tiles using mergetile.py')
    parser.add_argument('input_dir', type=str, help='Directory containing input tiles')
    parser.add_argument('output_dir', type=str, help='Directory to save merged tiles')
    return parser.parse_args()

def main():
    """
    Main function to merge tiles using the mergetile.py script.
    """
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tile_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for tile_file in tile_files:
        tile_path = os.path.join(input_dir, tile_file)
        command = ["python", "mergetile.py", tile_path, output_dir]
        subprocess.run(command)

if __name__ == "__main__":
    main()


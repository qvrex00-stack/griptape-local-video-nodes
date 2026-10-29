"""
Griptape Local Video Nodes - Install Script
Video Generation Nodes (MiniMax-H3, Wan2.1, AnimateDiff) & Video Gen Server (port 8090)

Usage:
    python install.py --griptape-dir "C:/Foundry/Griptape"
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

def copy_library(src_dir: Path, target_griptape_dir: Path):
    dest_lib = target_griptape_dir / "libraries" / "griptape-nodes-library-videogen"
    print(f">> Deploying video nodes to: {dest_lib}")
    dest_lib.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy griptape_nodes_library.json
    shutil.copy2(src_dir / "griptape_nodes_library.json", dest_lib / "griptape_nodes_library.json")
    
    # 2. Copy griptape_nodes_library package
    shutil.copytree(
        src_dir / "griptape_nodes_library",
        dest_lib / "griptape_nodes_library",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    print(">> Node library files copied successfully.")

    # 3. Copy video_gen_server if needed
    dest_server = target_griptape_dir / "video_gen_server"
    if not dest_server.exists():
        print(f">> Copying video_gen_server to: {dest_server}")
        shutil.copytree(
            src_dir / "video_gen_server",
            dest_server,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".venv", "outputs", "temp", "__pycache__", "*.pyc")
        )
    else:
        print(f">> Existing video_gen_server found at: {dest_server} (skipping overwrite)")

    print("\n✅ Installation complete! Restart Griptape to load the new video nodes.")

def main():
    parser = argparse.ArgumentParser(description="Install Griptape Local Video Nodes")
    parser.add_argument("--griptape-dir", default=r"C:\Foundry\Griptape", help="Griptape root directory")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    target_dir = Path(args.griptape_dir)

    if not target_dir.exists():
        print(f"Error: Target directory does not exist: {target_dir}")
        sys.exit(1)

    copy_library(src_dir, target_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os

from matplotlib.path import Path
from src.utils.config import load_config, get_dataset_path

cfg = load_config()

with open(str(Path(cfg["data_dir"]) / "ions.txt"), 'r') as f:
    ions = [line.strip() for line in f]


ion_directories = []

subdirs = []
for item in os.listdir(get_dataset_path(cfg, "ahojdb", "data")):
    if os.path.isdir(f'{get_dataset_path(cfg, "ahojdb", "data")}/{item}'):
        subdirs.append(item)

subdirs.sort()
print(f"Found {len(subdirs)} subdirectories to search")

for i, subdir in enumerate(subdirs):
    print(f"Checking {subdir} ({i+1}/{len(subdirs)})...")
    
    subdir_path = f'{get_dataset_path(cfg, "ahojdb", "data")}/{subdir}'
    
    try:
        entries = os.listdir(subdir_path)
        
        for entry_dir in entries:
            parts = entry_dir.split('-')
            if len(parts) >= 3:
                ligand = parts[2]
                
                if ligand in ions:
                    ion_directories.append(entry_dir)
                    
    except Exception as e:
        print(f"  Error in {subdir}: {e}")
        continue

print(f"\nTotal ion directories found: {len(ion_directories)}")

with open(str(Path(cfg["data_dir"]) / "all_ion_directories.txt"), 'w') as f:
    for dir_name in ion_directories:
        f.write(dir_name + '\n')
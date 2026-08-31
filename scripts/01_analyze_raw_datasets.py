import os
import hashlib
from pathlib import Path
from collections import defaultdict

raw_dir = Path("data/raw")
datasets = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

print("=== Raw Dataset Analysis ===\n")

for ds in datasets:
    print(f"Analyzing {ds.name}...")
    
    file_counts = defaultdict(int)
    name_collisions = defaultdict(list)
    size_map = defaultdict(list)
    
    # Traverse all files in the dataset
    total_files = 0
    for file_path in ds.rglob("*"):
        if file_path.is_file():
            total_files += 1
            ext = file_path.suffix.lower()
            file_counts[ext] += 1
            
            # Check for name collisions (same name, different subfolders)
            name_collisions[file_path.name].append(file_path)
            
            # Group by file size to quickly find potential exact duplicates
            size = file_path.stat().st_size
            size_map[size].append(file_path)

    # Find name collisions
    actual_name_collisions = {name: paths for name, paths in name_collisions.items() if len(paths) > 1}
    
    # Find exact byte-for-byte duplicates
    exact_duplicates = []
    for size, paths in size_map.items():
        if len(paths) > 1 and size > 0:
            # Hash them to be sure
            hashes = defaultdict(list)
            for p in paths:
                h = hashlib.md5(p.read_bytes()).hexdigest()
                hashes[h].append(p)
            for h, h_paths in hashes.items():
                if len(h_paths) > 1:
                    exact_duplicates.append(h_paths)
    
    print(f"  Total files: {total_files}")
    if file_counts:
        exts = ", ".join(f"{ext}: {count}" for ext, count in sorted(file_counts.items(), key=lambda x: -x[1])[:5])
        print(f"  Top extensions: {exts}")
    
    if actual_name_collisions:
        print(f"  [!] Found {len(actual_name_collisions)} filenames that appear multiple times in different subfolders.")
        # Print a couple examples
        examples = list(actual_name_collisions.items())[:3]
        for name, paths in examples:
            print(f"      Example '{name}': {len(paths)} occurrences")
    else:
        print("  [✓] No filename collisions found.")
        
    if exact_duplicates:
        print(f"  [!] Found {sum(len(d)-1 for d in exact_duplicates)} redundant identical files (exact byte match).")
    else:
        print("  [✓] No redundant identical files found.")
    
    print()


import pandas as pd
import numpy as np
import json
import os
import argparse

def find_clusters(coords, gap_threshold=0.5):
    """
    Find clustered intervals in a 1D coordinate array.
    """
    if len(coords) < 1:
        return []
    
    # Sort distinct coordinates
    unique_coords = np.sort(np.unique(coords))
    
    # Find indices where difference is larger than threshold
    diffs = np.diff(unique_coords)
    break_indices = np.where(diffs > gap_threshold)[0]
    
    # Extract clusters
    clusters = []
    start_idx = 0
    
    for end_idx in break_indices:
        # cluster from start_idx to end_idx (inclusive in unique_coords)
        c_min = unique_coords[start_idx]
        c_max = unique_coords[end_idx]
        clusters.append((c_min, c_max))
        start_idx = end_idx + 1
        
    # Last cluster
    c_min = unique_coords[start_idx]
    c_max = unique_coords[-1]
    clusters.append((c_min, c_max))
    
    return clusters

def calculate_grid_specs(csv_path, output_path, gap_threshold=0.5, z_min=0.4, fixed_x=None, fixed_y=None):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File {csv_path} not found.")
        return

    print(f"[INFO] Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out invalid points (using z > -0.5 as threshold, since voids seem to be -1.0)
    original_count = len(df)
    if 'z' in df.columns:
        df = df[df['z'] > z_min]
        print(f"[INFO] Filtered valid points: {len(df)}/{original_count} (kept Z > {z_min})")
    
    xs = df['x'].values
    ys = df['y'].values
    
    print(f"[INFO] Analyzing X axis distribution...")
    x_clusters = find_clusters(xs, gap_threshold)
    print(f"[INFO] Found {len(x_clusters)} clusters along X axis.")
    
    print(f"[INFO] Analyzing Y axis distribution...")
    y_clusters = find_clusters(ys, gap_threshold)
    print(f"[INFO] Found {len(y_clusters)} clusters along Y axis.")
    
    # Check if forced grid is requested
    if fixed_x is not None or fixed_y is not None:
        # Define ranges based on bounds
        x_min_all, x_max_all = xs.min(), xs.max()
        y_min_all, y_max_all = ys.min(), ys.max()

        # FORCE X
        if fixed_x and len(x_clusters) != fixed_x:
            print(f"[INFO] Cluster count {len(x_clusters)} != {fixed_x}. Forcing split X.")
            c_min, c_max = x_min_all, x_max_all
            step = (c_max - c_min) / fixed_x
            
            new_x_clusters = []
            for i in range(fixed_x):
                s_min = c_min + i*step
                s_max = c_min + (i+1)*step
                
                # Refine
                in_slot = xs[(xs >= s_min) & (xs <= s_max + 1e-5)]
                if len(in_slot) > 0:
                    new_x_clusters.append((in_slot.min(), in_slot.max()))
                else:
                    new_x_clusters.append((s_min, s_max))
            x_clusters = new_x_clusters

        # FORCE Y
        if fixed_y and len(y_clusters) != fixed_y:
            print(f"[INFO] Cluster count {len(y_clusters)} != {fixed_y}. Forcing split Y.")
            c_min, c_max = y_min_all, y_max_all
            step = (c_max - c_min) / fixed_y
            
            new_y_clusters = []
            for i in range(fixed_y):
                s_min = c_min + i*step
                s_max = c_min + (i+1)*step
                
                # Refine
                in_slot = ys[(ys >= s_min) & (ys <= s_max + 1e-5)]
                if len(in_slot) > 0:
                    new_y_clusters.append((in_slot.min(), in_slot.max()))
                else:
                    new_y_clusters.append((s_min, s_max))
            y_clusters = new_y_clusters
    
    # Validation/Redundancy check: if we already found the right number naturally, we keep them.
    # The above blocks only engage if clusters <= 1 (approx). 
    # Wait, if z_min=0.9 found 10 X clusters, we DON'T want to overwrite them with forced split unless necessary.
    # But if User explicitly asked for --num_x 10, maybe we should Ensure it is 10.
    # My logic above: `if len(x_clusters) <= 1 and fixed_x`. 
    # If I find 10 clusters and fixed_x is 10, I skip the block -> Good.
    # If I find 6 Y clusters and fixed_y is 15, I SKIP the block -> BAD.
    # I should change logic to: `if len(x_clusters) != fixed_x:`
    
    # Calculate dimensions
    x_lengths = [c[1] - c[0] for c in x_clusters]
    y_lengths = [c[1] - c[0] for c in y_clusters]
    
    if not x_lengths or not y_lengths:
        print("[ERROR] No clusters found! Adjust gap_threshold?")
        return

    avg_l = np.mean(x_lengths)
    avg_w = np.mean(y_lengths)
    
    print(f"\n[RESULT] Grid Structure: {len(x_clusters)} (X) x {len(y_clusters)} (Y)")
    print(f"[RESULT] Average Cell Size: L (X) ={avg_l:.4f}, W (Y) ={avg_w:.4f}")
    
    # Generate JSON
    cells = []
    
    # Sort clusters to ensure index matches geometric order (Left->Right, Bottom->Top)
    x_clusters.sort(key=lambda c: c[0])
    y_clusters.sort(key=lambda c: c[0])
    
    for i, x_c in enumerate(x_clusters):
        for j, y_c in enumerate(y_clusters):
            
            # Origin based on user request: 
            # "Select a corner point as origin, such that for any point inside (x,y), x>=origin_x and y>=origin_y"
            # This corresponds to the minimum X and minimum Y of the cell.
            x_min, x_max = x_c
            y_min, y_max = y_c
            
            origin_x = float(x_min)
            origin_y = float(y_min)
            
            cell_data = {
                "grid_index": [i, j],
                "origin": [origin_x, origin_y],
                "top_left": [origin_x, origin_y], # Legacy field for compatibility if needed, but 'origin' implies MinX, MinY. 
                                                # Warning: old code used (minX, maxY) as Top-Left. 
                                                # I will keep 'origin' as the specific request.
                "center": [float(x_min + x_max) / 2.0, float(y_min + y_max) / 2.0],
                "size": [float(x_max - x_min), float(y_max - y_min)],
                "bounds": {
                    "x_min": float(x_min), "x_max": float(x_max),
                    "y_min": float(y_min), "y_max": float(y_max)
                }
            }
            cells.append(cell_data)
            
    output_data = {
        "summary": {
            "num_cols_x": len(x_clusters),
            "num_rows_y": len(y_clusters),
            "avg_length_x": float(avg_l),
            "avg_width_y": float(avg_w),
            "gap_threshold_used": gap_threshold
        },
        "cells": cells
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"[INFO] Saved grid specs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaulting to terrain_height_map.csv as it seems to be the active one
    parser.add_argument("--csv", type=str, default="terrain_height_map.csv", help="Path to input terrain CSV")
    parser.add_argument("--out", type=str, default="grid_cells_top_left.json", help="Path to output JSON")
    parser.add_argument("--gap", type=float, default=1.0, help="Gap threshold to detect grid separation")
    parser.add_argument("--z_min", type=float, default=0.4, help="Min Z threshold for valid terrain")
    parser.add_argument("--num_x", type=int, default=None, help="Force number of cells in X")
    parser.add_argument("--num_y", type=int, default=None, help="Force number of cells in Y")
    args = parser.parse_args()
    
    calculate_grid_specs(args.csv, args.out, args.gap, args.z_min, args.num_x, args.num_y)

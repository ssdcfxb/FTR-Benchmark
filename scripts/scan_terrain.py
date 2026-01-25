# -*- coding: utf-8 -*-
"""
====================================
@File Name ：scan_terrain.py
@Description : Script to scan terrain height map and export to CSV.
====================================
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Scan and export terrain height map.")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task.")
parser.add_argument("--terrain", type=str, default="cur_mixed", help="Name of the terrain to use.")
parser.add_argument("--x_range", type=float, nargs=2, default=None, help="X coordinate range (min max). If None, full map is scanned.")
parser.add_argument("--y_range", type=float, nargs=2, default=None, help="Y coordinate range (min max). If None, full map is scanned.")
parser.add_argument("--step", type=float, default=0.1, help="Scanning step size (resolution) in meters.")
parser.add_argument("--native", action="store_true", default=False, help="Use native map resolution (0.05m). Ignores --step.")
parser.add_argument("--output", type=str, default="terrain_height_map.csv", help="Output CSV filename.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Override specific arguments for environment setup
sys.argv = [sys.argv[0]] + hydra_args + [
    f"env.scene.num_envs=1",  # Minimal envs needed for map reading
    f"env.terrain_name={args_cli.terrain}",
]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
import ftr_envs.tasks  # Register tasks

def main():
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu", # CPU is sufficient for map extraction
        num_envs=1
    )
    
    if hasattr(args_cli, "terrain"):
        env_cfg.terrain_name = args_cli.terrain
    
    # Create the environment
    print(f"[INFO] Creating environment to load terrain '{args_cli.terrain}'...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    try:
        # Access the underlying environment
        if hasattr(env, 'unwrapped'):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env

        # Check for terrain config
        if not hasattr(unwrapped_env, "terrain_cfg") or not hasattr(unwrapped_env.terrain_cfg, "map"):
            print("[ERROR] Environment does not have accessible terrain configuration or map.")
            return

        # Get map handler
        mh = unwrapped_env.terrain_cfg.map
        
        # Ensure map data is accessible (convert from torch if needed)
        map_data = mh.map
        if hasattr(map_data, "cpu"):
            map_data = map_data.cpu().numpy()
        elif hasattr(map_data, "numpy"):
            map_data = map_data.numpy()
            
        map_shape = map_data.shape
        print(f"[INFO] Terrain loaded. Map shape: {map_shape}, Cell size: {mh.cell_size}")

        map_shape = map_data.shape
        print(f"[INFO] Terrain loaded. Map shape: {map_shape}, Cell size: {mh.cell_size}")

        # Determine scanning bounds
        min_x_world = (0 - mh.compensation[0]) * mh.cell_size
        max_x_world = (map_shape[0] - mh.compensation[0]) * mh.cell_size
        min_y_world = (0 - mh.compensation[1]) * mh.cell_size
        max_y_world = (map_shape[1] - mh.compensation[1]) * mh.cell_size

        if args_cli.x_range is not None:
            scan_x_min, scan_x_max = args_cli.x_range
        else:
            scan_x_min, scan_x_max = min_x_world, max_x_world

        if args_cli.y_range is not None:
            scan_y_min, scan_y_max = args_cli.y_range
        else:
            scan_y_min, scan_y_max = min_y_world, max_y_world

        print(f"[INFO] Scanning Bounds: X[{scan_x_min:.2f}, {scan_x_max:.2f}], Y[{scan_y_min:.2f}, {scan_y_max:.2f}]")

        results = []

        if args_cli.native:
            print("[INFO] Mode: NATIVE resolution (Grid Index Iteration)")
            # Calculate grid indices from world bounds
            start_px = int(np.floor(scan_x_min / mh.cell_size + mh.compensation[0]))
            end_px = int(np.ceil(scan_x_max / mh.cell_size + mh.compensation[0]))
            start_py = int(np.floor(scan_y_min / mh.cell_size + mh.compensation[1]))
            end_py = int(np.ceil(scan_y_max / mh.cell_size + mh.compensation[1]))
            
            # Clamp to valid map range
            start_px = max(0, start_px)
            end_px = min(map_shape[0], end_px)
            start_py = max(0, start_py)
            end_py = min(map_shape[1], end_py)
            
            print(f"[INFO] Grid Index Range: X[{start_px}:{end_px}], Y[{start_py}:{end_py}]")
            print(f"[INFO] Total points to scan: {(end_px - start_px) * (end_py - start_py)}")

            for px in range(start_px, end_px):
                x = (px - mh.compensation[0]) * mh.cell_size
                for py in range(start_py, end_py):
                    # Direct grid access
                    height = float(map_data[px, py])
                    y = (py - mh.compensation[1]) * mh.cell_size
                    results.append({"x": x, "y": y, "z": height})
        
        else:
            print(f"[INFO] Mode: CUSTOM resolution (Step: {args_cli.step}m)")
            x_values = np.arange(scan_x_min, scan_x_max + args_cli.step, args_cli.step)
            y_values = np.arange(scan_y_min, scan_y_max + args_cli.step, args_cli.step)
            print(f"[INFO] Total points to scan: {len(x_values) * len(y_values)}")

            for x in x_values:
                for y in y_values:
                    # Bilinear Interpolation for smoother high-res sampling
                    u = (x / mh.cell_size) + mh.compensation[0]
                    v = (y / mh.cell_size) + mh.compensation[1]
                    
                    px = int(np.floor(u))
                    py = int(np.floor(v))
                    
                    height = -100.0 # Default out of bounds
                    
                    # Check if we can interpolate (needs px+1, py+1)
                    if 0 <= px < map_shape[0] - 1 and 0 <= py < map_shape[1] - 1:
                        rx = u - px
                        ry = v - py
                        
                        h00 = float(map_data[px, py])
                        h10 = float(map_data[px+1, py])
                        h01 = float(map_data[px, py+1])
                        h11 = float(map_data[px+1, py+1])
                        
                        # Interpolate X
                        h0 = h00 * (1 - rx) + h10 * rx
                        h1 = h01 * (1 - rx) + h11 * rx
                        # Interpolate Y
                        height = h0 * (1 - ry) + h1 * ry
                    elif 0 <= px < map_shape[0] and 0 <= py < map_shape[1]:
                        # Edge pixels (no neighbors for interpolation)
                        height = float(map_data[px, py])

                    results.append({"x": x, "y": y, "z": height})
        
        # Export to CSV
        
        # Export to CSV
        print(f"[INFO] Exporting data to {args_cli.output}...")
        df = pd.DataFrame(results)
        df.to_csv(args_cli.output, index=False)
        print(f"[SUCCESS] Done.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] An error occurred: {e}")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()

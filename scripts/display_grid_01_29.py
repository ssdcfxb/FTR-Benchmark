# -*- coding: utf-8 -*-
"""
====================================
@File Name ：display_grid.py
@Description : Script to spawn multiple robots on a grid based on terrain map, iterating through various initial states.
====================================
"""

import argparse
import sys
import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import traceback
from omni.isaac.lab.app import AppLauncher

import json

# add argparse arguments
parser = argparse.ArgumentParser(description="Grid search sampling of robot stable states on various terrains.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate (number of robots per batch).")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task.")
parser.add_argument("--terrain", type=str, default="cur_mixed", help="Name of the terrain to use (e.g., cur_mixed, cur_stairs_up).")
parser.add_argument("--spawn_height", type=float, default=0.8, help="Height to spawn robots above the ground.")
parser.add_argument("--settle_steps", type=int, default=40, help="Number of steps to wait for robots to settle.")
parser.add_argument("--sample_steps", type=int, default=20, help="Number of steps to record after settling.")
parser.add_argument("--output_dir", type=str, default="logs/data_collection_grid", help="Directory to save collected data.")
parser.add_argument("--grid_json", type=str, default="grid_cells_top_left.json", help="Path to the grid specifications JSON file.")
parser.add_argument("--k_rounds", type=int, default=1, help="Number of rounds to sample for each parameter combination.")

# Grid search parameters (Overridden by JSON for XY if used)
parser.add_argument("--grid_res_x", type=float, default=5.0, help="Grid resolution for X in meters (if not using JSON).")
parser.add_argument("--grid_res_y", type=float, default=5.0, help="Grid resolution for Y in meters (if not using JSON).")
parser.add_argument("--grid_res_yaw", type=float, default=45.0, help="Resolution for Yaw in degrees.")
parser.add_argument("--grid_res_flipper", type=float, default=10.0, help="Resolution for flipper angle in degrees.")
parser.add_argument("--flipper_range", type=float, default=70.0, help="Range for flipper angle (+/- degrees).")

# Randomization parameters
parser.add_argument("--rand_xy", type=float, default=0.5, help="Random bias range for XY position in meters.")
parser.add_argument("--rand_yaw", type=float, default=5.0, help="Random bias range for Yaw in degrees.")
parser.add_argument("--rand_flipper", type=float, default=5.0, help="Random bias range for flipper angle in degrees.")


AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Override specific arguments for environment setup
sys.argv = [sys.argv[0]] + hydra_args + [
    f"env.scene.num_envs={args_cli.num_envs}",
    f"env.terrain_name={args_cli.terrain}",
]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import quat_from_euler_xyz
import ftr_envs.tasks  # Register tasks
from ftr_envs.tasks.crossing.ftr_env import FtrEnv

def load_grid_from_json(json_path):
    """
    Loads grid data from JSON file. Returns list of cell dicts.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Grid JSON file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Validating grid data
    if isinstance(data, list):
        # Support for direct list of bounds (e.g. refined_grid_cells_bounds.json)
        # Adapt to expected format with 'center' and 'origin'
        cells = []
        for idx, item in enumerate(data):
            # refined_grid_cells_bounds.json structure: {x_min, x_max, y_min, y_max}
            if 'x_min' in item and 'x_max' in item and 'y_min' in item and 'y_max' in item:
                x_min, x_max = item['x_min'], item['x_max']
                y_min, y_max = item['y_min'], item['y_max']
                cells.append({
                    "grid_index": [idx, 0], # Placeholder index
                    "origin": [x_min, y_min],
                    "center": [(x_min + x_max)/2.0, (y_min + y_max)/2.0],
                    "size": [x_max - x_min, y_max - y_min],
                    "bounds": item
                })
            else:
                # Assuming it might already be in correct format
                cells.append(item)
    else:
        # Standard format { "cells": [...] }
        cells = data.get("cells", [])
        
    print(f"[INFO] Loaded {len(cells)} grid cells from {json_path}.")
    return cells

def get_terrain_height(x, y, terrain_map, cell_size=0.05):
    """
    Look up terrain height from the map (simple nearest neighbor or pre-loaded dataframe lookup).
    For efficiency with large CSVs during runtime, we rely on the simulator's terrain if possible,
    but here we might need initial estimates.
    """
    # Note: To be perfectly accurate, we should query the simulation mesh or heightfield.
    # The provided CSV lookup might be slow if done per point naively.
    # For now, we will trust the simulation to handle ground collision 
    # and just spawn slightly above the estimated max height in that cell.
    return 0.0 # Placeholder, will rely on sim raycast or height field query if needed, 
               # OR easier: spawn high enough and let it fall.

def main():
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu" if args_cli.cpu else "cuda:0",
        num_envs=args_cli.num_envs
    )
    
    if hasattr(args_cli, "terrain"):
        env_cfg.terrain_name = args_cli.terrain
    
    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    print(f"[INFO] Environment created with {args_cli.num_envs} instances.")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args_cli.output_dir, f"{args_cli.terrain}_grid_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    csv_file = os.path.join(save_path, "robot_states_grid.csv")
    first_write = True

    try:
        unwrapped_env = env.unwrapped
        robot = unwrapped_env._robot
        
        # 1. Load Map & Grid Generation
        grid_cells = load_grid_from_json(args_cli.grid_json)
        if not grid_cells:
            return

        # 2. Define Parameter Ranges
        yaw_values = np.arange(0, 360, args_cli.grid_res_yaw)
        flipper_values = -np.arange(-args_cli.flipper_range, args_cli.flipper_range + args_cli.grid_res_flipper, args_cli.grid_res_flipper)
        
        # Total combinations estimate
        total_grids = len(grid_cells)
        total_yaws = len(yaw_values)
        total_flippers = len(flipper_values)
        total_simulations = total_grids * total_yaws * total_flippers * total_flippers * args_cli.k_rounds
        print(f"[INFO] Search Space: {total_grids} Locs * {total_yaws} Yaws * {total_flippers} Front * {total_flippers} Rear * {args_cli.k_rounds} Rounds = {total_simulations} Total Samples")
        
        # We process parameter combinations in batches.
        # Structure: Outer loops iterate grid cells, yaw, flippers.
        # Inner loop (k_rounds) applies randomization to a nominal state.
        
        current_batch_configs = []
        batch_idx = 0
        global_sample_id = 0
        
        # 3. Loops - Inverted Order to maximize spatial diversity in each batch
        # New Order: Yaw -> Theta_Front -> Theta_Rear -> K_Round -> Grid
        # This ensures that consecutive tasks are in different grid cells.
        for f_front_base in flipper_values:
            for f_rear_base in flipper_values:
                for yaw_base in yaw_values:
                    for k in range(args_cli.k_rounds):
                        for cell in grid_cells:
                            
                            # Use Center as the base spawn point
                            base_x = cell.get("center", [0,0])[0]
                            base_y = cell.get("center", [0,0])[1]
                            origin_x = cell.get("origin", [0,0])[0]
                            origin_y = cell.get("origin", [0,0])[1]

                            # Prepare config for one robot instance
                            cfg = {
                                "sample_id": global_sample_id,
                                "base_x": base_x, 
                                "base_y": base_y,
                                "origin_x": origin_x,
                                "origin_y": origin_y,
                                "size_x": cell.get("size", [None, None])[0],
                                "size_y": cell.get("size", [None, None])[1],
                                "base_yaw": yaw_base,
                                "base_front": f_front_base,
                                "base_rear": f_rear_base,
                                "k_round": k
                            }
                            global_sample_id += 1
                            
                            current_batch_configs.append(cfg)
                            
                            # If batch is full, execute simulation
                            if len(current_batch_configs) >= args_cli.num_envs:
                                process_batch(env, unwrapped_env, robot, current_batch_configs, args_cli, csv_file, save_header=first_write)
                                first_write = False
                                current_batch_configs = []
                                batch_idx += 1
                                progress_pct = (global_sample_id / total_simulations) * 100
                                print(f"[INFO] Processed Batch {batch_idx} | Progress: {global_sample_id}/{total_simulations} ({progress_pct:.2f}%)")

        # Process remaining
        if len(current_batch_configs) > 0:
            process_batch(env, unwrapped_env, robot, current_batch_configs, args_cli, csv_file, save_header=first_write)
            batch_idx += 1
            progress_pct = (global_sample_id / total_simulations) * 100
            print(f"[INFO] Processed Final Batch {batch_idx} | Progress: {global_sample_id}/{total_simulations} ({progress_pct:.2f}%)")

    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] An error occurred: {e}")
    finally:
        env.close()
        simulation_app.close()

def process_batch(env, unwrapped_env, robot, configs, args, csv_file, save_header=False):
    """
    Executes one simulation batch for the given configurations.
    """
    num_robots = len(configs)
    # If partial batch, we only care about the first num_robots envs, 
    # but we must simulate all num_envs to satisfy gym. We'll just ignore output of unused ones.
    
    # Reset Environment
    env.reset()
    
    # Prepare State Arrays
    root_state = robot.data.root_state_w.clone()
    flipper_targets = torch.zeros((env.num_envs, 4), device=unwrapped_env.device)
    
    # Store initial params for recording
    batch_records = []
    
    for i in range(num_robots):
        cfg = configs[i]
        
        # Apply Random Bias
        # bias ranges: xy +/- rand_xy, yaw +/- rand_yaw, flipper +/- rand_flipper
        
        # Position
        # Use specific cell size for range if available, otherwise use default rand_xy
        range_x = cfg.get("size_x") / 2.0 if cfg.get("size_x") is not None else args.rand_xy
        range_y = cfg.get("size_y") / 2.0 if cfg.get("size_y") is not None else args.rand_xy

        bias_x = np.random.uniform(-range_x, range_x)
        bias_y = np.random.uniform(-range_y, range_y)
        
        pos_x = cfg["base_x"] + bias_x
        pos_y = cfg["base_y"] + bias_y
        
        # Enforce Constraint: x, y >= origin
        # pos_x = max(pos_x, cfg["origin_x"])
        # pos_y = max(pos_y, cfg["origin_y"])
        
        # Get terrain height at this specific x,y
        # We use the terrain map in the env to verify height
        ground_z = 0.0
        if hasattr(unwrapped_env, "terrain_cfg") and hasattr(unwrapped_env.terrain_cfg, "map"):
            mh = unwrapped_env.terrain_cfg.map
            # Convert world to pixel
            px = int(np.floor((pos_x) / mh.cell_size + mh.compensation[0]))
            py = int(np.floor((pos_y) / mh.cell_size + mh.compensation[1]))
            # Safe crop
            px = np.clip(px, 0, mh.map.shape[0]-1)
            py = np.clip(py, 0, mh.map.shape[1]-1)
            ground_z = float(mh.map[px, py])
            
        pos_z = ground_z + args.spawn_height
        
        # Orientation (Yaw)
        yaw_deg = cfg["base_yaw"] + np.random.uniform(-args.rand_yaw, args.rand_yaw)
        yaw_rad = np.deg2rad(yaw_deg)
        # r, p, y -> quat
        # Assuming flat start (roll=0, pitch=0)
        q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw_rad])).squeeze()
        
        # Flipper Angles
        f_front_deg = cfg["base_front"] + np.random.uniform(-args.rand_flipper, args.rand_flipper)
        f_rear_deg = cfg["base_rear"] + np.random.uniform(-args.rand_flipper, args.rand_flipper)
        
        # Set Root State
        root_state[i, 0] = pos_x
        root_state[i, 1] = pos_y
        root_state[i, 2] = pos_z
        root_state[i, 3:7] = q.to(root_state.device) # w, x, y, z
        root_state[i, 7:13] = 0.0 # Zero velocities
        
        # Set Flipper Tensor
        # 0:FL, 1:FR, 2:RL, 3:RR
        flipper_targets[i, 0] = f_front_deg
        flipper_targets[i, 1] = f_front_deg
        flipper_targets[i, 2] = f_rear_deg
        flipper_targets[i, 3] = f_rear_deg
        
        # Record Bias / Init Info
        batch_records.append({
            "sample_id": cfg["sample_id"],
            "robot_id": i,
            "base_x": cfg["base_x"],
            "base_y": cfg["base_y"],
            "base_yaw": cfg["base_yaw"],
            "base_front": cfg["base_front"],
            "base_rear": cfg["base_rear"],
            "k_round": cfg["k_round"],
            "init_x": pos_x,
            "init_y": pos_y,
            "init_z": pos_z,
            "init_roll": 0.0,
            "init_pitch": 0.0,
            "init_yaw": yaw_deg,
            "init_front": f_front_deg,
            "init_rear": f_rear_deg,
            "init_qx": float(q[1]),
            "init_qy": float(q[2]),
            "init_qz": float(q[3]),
            "init_qw": float(q[0]),
            "bias_x": pos_x - cfg["base_x"],
            "bias_y": pos_y - cfg["base_y"],
            "bias_yaw": yaw_deg - cfg["base_yaw"],
            "bias_front": f_front_deg - cfg["base_front"],
            "bias_rear": f_rear_deg - cfg["base_rear"]
        })

    # Write Root State
    robot.write_root_state_to_sim(root_state, env_ids=torch.arange(env.num_envs, device=unwrapped_env.device))
    
    # Write Flipper Pos
    robot.set_all_flipper_positions(flipper_targets, degree=True)
    
    # Init Env Command Targets (for action application)
    unwrapped_env.forward_vel_commands[:] = 0.0
    unwrapped_env.flipper_target_pos[:] = 0.0
    # Copy degrees to rads for target
    flipper_rads = torch.deg2rad(flipper_targets)
    if unwrapped_env.flipper_target_pos.shape[1] >= 4:
        unwrapped_env.flipper_target_pos[:, :] = flipper_rads[:, :]
    
    unwrapped_env.suppress_done_signals = True

    # Settle
    lin_vel_threshold = 0.01
    ang_vel_threshold = 0.01
    
    for _ in range(args.settle_steps):
        actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
        env.step(actions)
        
    # Sample (Record last 5 frames)
    steps_to_record = 5
    pre_record_steps = max(0, args.sample_steps - steps_to_record)
    
    # 1. Run simulation without recording
    for _ in range(pre_record_steps):
        actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
        env.step(actions)
    
    # 2. Run simulation WITH recording
    data_list = []
    
    for step_idx in range(steps_to_record):
        actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
        env.step(actions)
        
        lin_vel = unwrapped_env.robot_lin_velocities.cpu().numpy()
        ang_vel = unwrapped_env.robot_ang_velocities.cpu().numpy()
        
        # Stability Check
        lin_norm = np.linalg.norm(lin_vel, axis=1)
        ang_norm = np.linalg.norm(ang_vel, axis=1)
        stable_mask = (lin_norm < lin_vel_threshold) & (ang_norm < ang_vel_threshold)
        
        # Capture Current State
        cur_pos = unwrapped_env.positions.cpu().numpy()
        cur_quat = unwrapped_env.orientations.cpu().numpy() # w, x, y, z
        cur_joints = unwrapped_env.flipper_positions.cpu().numpy()
        
        for i in range(num_robots):
            rec = batch_records[i].copy()

            # Stability Check (Abnormal)
            # Reference from display.py, but relaxed Z bounds for terrain grid search
            is_abnormal = (
                np.isnan(cur_pos[i]).any() or 
                np.isnan(lin_vel[i]).any() or
                np.isnan(ang_vel[i]).any() or
                lin_norm[i] > 10.0 or
                ang_norm[i] > 10.0 or
                cur_pos[i, 2] < -5.0 or     # Fallen off world (relaxed from -5.0)
                cur_pos[i, 2] > 10.0        # Flown away (relaxed from 5.0)
            )
            
            # Expand record with current frame state
            rec.update({
                "record_frame_index": step_idx, # 0 to 4
                "is_abnormal": int(is_abnormal),
                "final_x": float(cur_pos[i, 0]),
                "final_y": float(cur_pos[i, 1]),
                "final_z": float(cur_pos[i, 2]),
                "final_qw": float(cur_quat[i, 0]),
                "final_qx": float(cur_quat[i, 1]),
                "final_qy": float(cur_quat[i, 2]),
                "final_qz": float(cur_quat[i, 3]),
                "final_flipper_0": float(cur_joints[i, 0]),
                "final_flipper_1": float(cur_joints[i, 1]),
                "final_flipper_2": float(cur_joints[i, 2]),
                "final_flipper_3": float(cur_joints[i, 3]),
                "final_lin_vel": float(np.linalg.norm(lin_vel[i])),
                "final_ang_vel": float(np.linalg.norm(ang_vel[i])),
                "is_stable": int(stable_mask[i])
            })
            data_list.append(rec)
        
    # Save to CSV
    df = pd.DataFrame(data_list)
    df.to_csv(csv_file, mode='a', header=save_header, index=False)
    
    # Explicit garbage collection to prevent memory buildup
    del df
    del data_list
    import gc
    gc.collect()

if __name__ == "__main__":
    main()


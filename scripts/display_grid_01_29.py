# -*- coding: utf-8 -*-
"""
====================================
@File Name ：display_grid_parquet_fast.py
@Description : Fast grid search sampling + Parquet output (partitioned by batch).
- Vectorized init randomization (XY/yaw/flippers)
- Reuse action tensor
- Vectorized per-frame logging (no per-robot dict copy)
- Write one parquet per batch (no expensive CSV append)
====================================
"""

import argparse
import sys
import os
from datetime import datetime
import traceback
import json

import torch
import numpy as np
import pandas as pd

from omni.isaac.lab.app import AppLauncher

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

# Parquet-specific
parser.add_argument("--parquet_compression", type=str, default="snappy", help="Parquet compression: snappy/zstd/gzip/none.")
parser.add_argument("--record_last_frames", type=int, default=5, help="How many final frames to record per batch (default 5).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# override env scene
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
import ftr_envs.tasks  # noqa: F401

def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception as e:
        raise RuntimeError(
            "Writing Parquet requires 'pyarrow'. Install with:\n"
            "  pip install pyarrow\n"
            f"Original import error: {repr(e)}"
        )


def load_grid_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"[ERROR] Grid JSON file not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        cells = []
        for idx, item in enumerate(data):
            if "x_min" in item and "x_max" in item and "y_min" in item and "y_max" in item:
                x_min, x_max = item["x_min"], item["x_max"]
                y_min, y_max = item["y_min"], item["y_max"]
                cells.append(
                    {
                        "grid_index": [idx, 0],
                        "origin": [x_min, y_min],
                        "center": [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
                        "size": [x_max - x_min, y_max - y_min],
                        "bounds": item,
                    }
                )
            else:
                cells.append(item)
    else:
        cells = data.get("cells", [])

    print(f"[INFO] Loaded {len(cells)} grid cells from {json_path}.")
    return cells


def _cfg_arrays(configs):
    """Extract per-robot cfg arrays from list[dict]."""
    n = len(configs)
    sample_id = np.empty(n, dtype=np.int64)
    base_x = np.empty(n, dtype=np.float64)
    base_y = np.empty(n, dtype=np.float64)
    origin_x = np.empty(n, dtype=np.float64)
    origin_y = np.empty(n, dtype=np.float64)

    size_x = np.empty(n, dtype=np.float64)
    size_y = np.empty(n, dtype=np.float64)
    size_x.fill(np.nan)
    size_y.fill(np.nan)

    base_yaw = np.empty(n, dtype=np.float64)
    base_front = np.empty(n, dtype=np.float64)
    base_rear = np.empty(n, dtype=np.float64)
    k_round = np.empty(n, dtype=np.int64)

    for i, c in enumerate(configs):
        sample_id[i] = c["sample_id"]
        base_x[i] = c["base_x"]
        base_y[i] = c["base_y"]
        origin_x[i] = c["origin_x"]
        origin_y[i] = c["origin_y"]

        sx = c.get("size_x", None)
        sy = c.get("size_y", None)
        if sx is not None:
            size_x[i] = sx
        if sy is not None:
            size_y[i] = sy

        base_yaw[i] = c["base_yaw"]
        base_front[i] = c["base_front"]
        base_rear[i] = c["base_rear"]
        k_round[i] = c["k_round"]

    return (
        sample_id,
        base_x,
        base_y,
        origin_x,
        origin_y,
        size_x,
        size_y,
        base_yaw,
        base_front,
        base_rear,
        k_round,
    )


def process_batch_parquet_fast(env, unwrapped_env, robot, configs, args, parquet_dir, batch_idx):
    """
    Executes one simulation batch for the given configurations.
    Writes ONE parquet file per batch: part-000000.parquet, part-000001.parquet, ...
    """
    num_robots = len(configs)
    env.reset()

    device = unwrapped_env.device
    action_dim = unwrapped_env.action_space.shape[1]
    actions = torch.zeros((env.num_envs, action_dim), device=device)  # reuse

    root_state = robot.data.root_state_w.clone()
    flipper_targets = torch.zeros((env.num_envs, 4), device=device)

    # -------- vectorized init sampling --------
    (
        sample_id,
        base_x,
        base_y,
        origin_x,
        origin_y,
        size_x,
        size_y,
        base_yaw,
        base_front,
        base_rear,
        k_round,
    ) = _cfg_arrays(configs)

    range_x = np.where(np.isfinite(size_x), size_x / 2.0, args.rand_xy)
    range_y = np.where(np.isfinite(size_y), size_y / 2.0, args.rand_xy)

    bias_x = (np.random.rand(num_robots) * 2.0 - 1.0) * range_x
    bias_y = (np.random.rand(num_robots) * 2.0 - 1.0) * range_y
    pos_x = base_x + bias_x
    pos_y = base_y + bias_y

    # terrain height (vectorized) using terrain_cfg.map
    mh = unwrapped_env.terrain_cfg.map
    px = np.floor(pos_x / mh.cell_size + mh.compensation[0]).astype(np.int64)
    py = np.floor(pos_y / mh.cell_size + mh.compensation[1]).astype(np.int64)
    px = np.clip(px, 0, mh.map.shape[0] - 1)
    py = np.clip(py, 0, mh.map.shape[1] - 1)
    ground_z = mh.map[px, py].astype(np.float64)

    pos_z = ground_z + args.spawn_height

    yaw_deg = base_yaw + (np.random.rand(num_robots) * 2.0 - 1.0) * args.rand_yaw
    yaw_rad = np.deg2rad(yaw_deg).astype(np.float32)

    zeros = torch.zeros((num_robots,), device=device, dtype=torch.float32)
    yaw_t = torch.from_numpy(yaw_rad).to(device=device)
    q = quat_from_euler_xyz(zeros, zeros, yaw_t)  # (N,4) w,x,y,z

    f_front_deg = base_front + (np.random.rand(num_robots) * 2.0 - 1.0) * args.rand_flipper
    f_rear_deg = base_rear + (np.random.rand(num_robots) * 2.0 - 1.0) * args.rand_flipper

    root_state[:num_robots, 0] = torch.from_numpy(pos_x).to(device=device, dtype=root_state.dtype)
    root_state[:num_robots, 1] = torch.from_numpy(pos_y).to(device=device, dtype=root_state.dtype)
    root_state[:num_robots, 2] = torch.from_numpy(pos_z).to(device=device, dtype=root_state.dtype)
    root_state[:num_robots, 3:7] = q.to(dtype=root_state.dtype)
    root_state[:num_robots, 7:13] = 0.0

    ff = torch.from_numpy(f_front_deg.astype(np.float32)).to(device=device)
    fr = torch.from_numpy(f_rear_deg.astype(np.float32)).to(device=device)
    flipper_targets[:num_robots, 0] = ff
    flipper_targets[:num_robots, 1] = ff
    flipper_targets[:num_robots, 2] = fr
    flipper_targets[:num_robots, 3] = fr

    robot.write_root_state_to_sim(root_state, env_ids=torch.arange(env.num_envs, device=device))
    robot.set_all_flipper_positions(flipper_targets, degree=True)

    unwrapped_env.forward_vel_commands[:] = 0.0
    unwrapped_env.flipper_target_pos[:] = 0.0
    unwrapped_env.flipper_target_pos[:, :] = torch.deg2rad(flipper_targets)

    unwrapped_env.suppress_done_signals = True

    # -------- settle + pre_record --------
    for _ in range(args.settle_steps):
        env.step(actions)

    steps_to_record = int(args.record_last_frames)
    pre_record_steps = max(0, args.sample_steps - steps_to_record)
    for _ in range(pre_record_steps):
        env.step(actions)

    # -------- record last frames --------
    lin_vel_threshold = 0.01
    ang_vel_threshold = 0.01

    robot_id = np.arange(num_robots, dtype=np.int64)
    init_roll = np.zeros(num_robots, dtype=np.float64)
    init_pitch = np.zeros(num_robots, dtype=np.float64)

    q_cpu = q.detach().cpu().numpy()  # (N,4) w,x,y,z
    init_qw = q_cpu[:, 0].astype(np.float64)
    init_qx = q_cpu[:, 1].astype(np.float64)
    init_qy = q_cpu[:, 2].astype(np.float64)
    init_qz = q_cpu[:, 3].astype(np.float64)

    bias_yaw = yaw_deg - base_yaw
    bias_front = f_front_deg - base_front
    bias_rear = f_rear_deg - base_rear

    frames = []
    for step_idx in range(steps_to_record):
        env.step(actions)

        lin_vel = unwrapped_env.robot_lin_velocities[:num_robots].detach().cpu().numpy()
        ang_vel = unwrapped_env.robot_ang_velocities[:num_robots].detach().cpu().numpy()
        cur_pos = unwrapped_env.positions[:num_robots].detach().cpu().numpy()
        cur_quat = unwrapped_env.orientations[:num_robots].detach().cpu().numpy()  # w,x,y,z
        cur_joints = unwrapped_env.flipper_positions[:num_robots].detach().cpu().numpy()

        lin_norm = np.linalg.norm(lin_vel, axis=1)
        ang_norm = np.linalg.norm(ang_vel, axis=1)
        stable_mask = (lin_norm < lin_vel_threshold) & (ang_norm < ang_vel_threshold)

        is_nan = np.isnan(cur_pos).any(axis=1) | np.isnan(lin_vel).any(axis=1) | np.isnan(ang_vel).any(axis=1)
        is_abnormal = (
            is_nan
            | (lin_norm > 10.0)
            | (ang_norm > 10.0)
            | (cur_pos[:, 2] < -5.0)
            | (cur_pos[:, 2] > 10.0)
        )

        frame = {
            "sample_id": sample_id,
            "robot_id": robot_id,
            "base_x": base_x,
            "base_y": base_y,
            "base_yaw": base_yaw,
            "base_front": base_front,
            "base_rear": base_rear,
            "k_round": k_round,
            "init_x": pos_x,
            "init_y": pos_y,
            "init_z": pos_z,
            "init_roll": init_roll,
            "init_pitch": init_pitch,
            "init_yaw": yaw_deg,
            "init_front": f_front_deg,
            "init_rear": f_rear_deg,
            "init_qx": init_qx,
            "init_qy": init_qy,
            "init_qz": init_qz,
            "init_qw": init_qw,
            "bias_x": bias_x,
            "bias_y": bias_y,
            "bias_yaw": bias_yaw,
            "bias_front": bias_front,
            "bias_rear": bias_rear,
            "record_frame_index": np.full(num_robots, step_idx, dtype=np.int64),
            "is_abnormal": is_abnormal.astype(np.int8),
            "final_x": cur_pos[:, 0].astype(np.float32),
            "final_y": cur_pos[:, 1].astype(np.float32),
            "final_z": cur_pos[:, 2].astype(np.float32),
            "final_qw": cur_quat[:, 0].astype(np.float32),
            "final_qx": cur_quat[:, 1].astype(np.float32),
            "final_qy": cur_quat[:, 2].astype(np.float32),
            "final_qz": cur_quat[:, 3].astype(np.float32),
            "final_flipper_0": cur_joints[:, 0].astype(np.float32),
            "final_flipper_1": cur_joints[:, 1].astype(np.float32),
            "final_flipper_2": cur_joints[:, 2].astype(np.float32),
            "final_flipper_3": cur_joints[:, 3].astype(np.float32),
            "final_lin_vel": lin_norm.astype(np.float32),
            "final_ang_vel": ang_norm.astype(np.float32),
            "is_stable": stable_mask.astype(np.int8),
        }
        frames.append(pd.DataFrame(frame))

    df_batch = pd.concat(frames, ignore_index=True)

    # -------- write parquet shard --------
    part_path = os.path.join(parquet_dir, f"part-{batch_idx:06d}.parquet")
    compression = None if args.parquet_compression.lower() in ("none", "no", "false") else args.parquet_compression
    df_batch.to_parquet(part_path, engine="pyarrow", compression=compression, index=False)




def main():
    _require_pyarrow()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu" if args_cli.cpu else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    if hasattr(args_cli, "terrain"):
        env_cfg.terrain_name = args_cli.terrain
    
    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    print(f"[INFO] Environment created with {args_cli.num_envs} instances.")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args_cli.output_dir, f"{args_cli.terrain}_grid_state_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    parquet_dir = os.path.join(save_path, "parquet_parts")
    os.makedirs(parquet_dir, exist_ok=True)

    meta_path = os.path.join(save_path, "meta.json")

    try:
        unwrapped_env = env.unwrapped
        robot = unwrapped_env._robot

        grid_cells = load_grid_from_json(args_cli.grid_json)
        if not grid_cells:
            return

        yaw_values = np.arange(0, 360, args_cli.grid_res_yaw)
        flipper_values = -np.arange(-args_cli.flipper_range,
                                    args_cli.flipper_range + args_cli.grid_res_flipper,
                                    args_cli.grid_res_flipper)

        total_grids = len(grid_cells)
        total_yaws = len(yaw_values)
        total_flippers = len(flipper_values)
        total_simulations = total_grids * total_yaws * total_flippers * total_flippers * args_cli.k_rounds
        print(
            f"[INFO] Search Space: {total_grids} Locs * {total_yaws} Yaws * "
            f"{total_flippers} Front * {total_flippers} Rear * {args_cli.k_rounds} Rounds "
            f"= {total_simulations} Total Samples"
        )

        # metadata (useful for reading later)
        meta = {
            "terrain": args_cli.terrain,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "grid_cells": total_grids,
            "yaw_values": yaw_values.tolist(),
            "flipper_values": flipper_values.tolist(),
            "k_rounds": int(args_cli.k_rounds),
            "settle_steps": int(args_cli.settle_steps),
            "sample_steps": int(args_cli.sample_steps),
            "record_last_frames": int(args_cli.record_last_frames),
            "parquet_dir": os.path.basename(parquet_dir),
            "parquet_compression": args_cli.parquet_compression,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        current_batch_configs = []
        batch_idx = 0
        global_sample_id = 0

        # Loop order kept consistent with your script:
        # f_front -> f_rear -> yaw -> k_round -> grid
        for k in range(args_cli.k_rounds):
            for f_front_base in flipper_values:
                for f_rear_base in flipper_values:
                    for yaw_base in yaw_values:
                        for cell in grid_cells:
                            base_x = cell.get("center", [0, 0])[0]
                            base_y = cell.get("center", [0, 0])[1]
                            origin_x = cell.get("origin", [0, 0])[0]
                            origin_y = cell.get("origin", [0, 0])[1]
                            size = cell.get("size", [None, None])

                            cfg = {
                                "sample_id": global_sample_id,
                                "base_x": base_x,
                                "base_y": base_y,
                                "origin_x": origin_x,
                                "origin_y": origin_y,
                                "size_x": size[0],
                                "size_y": size[1],
                                "base_yaw": yaw_base,
                                "base_front": f_front_base,
                                "base_rear": f_rear_base,
                                "k_round": k,
                            }
                            global_sample_id += 1
                            current_batch_configs.append(cfg)

                            if len(current_batch_configs) >= args_cli.num_envs:
                                process_batch_parquet_fast(
                                    env, unwrapped_env, robot,
                                    current_batch_configs, args_cli,
                                    parquet_dir, batch_idx
                                )
                                current_batch_configs = []
                                batch_idx += 1

                                progress_pct = (global_sample_id / total_simulations) * 100.0
                                print(
                                    f"[INFO] Wrote Batch {batch_idx} | Progress: {global_sample_id}/{total_simulations} ({progress_pct:.2f}%)"
                                )

        if len(current_batch_configs) > 0:
            process_batch_parquet_fast(env, unwrapped_env, robot, current_batch_configs, args_cli, parquet_dir, batch_idx)
            batch_idx += 1
            progress_pct = (global_sample_id / total_simulations) * 100.0
            print(f"[INFO] Wrote Final Batch {batch_idx} | Progress: {global_sample_id}/{total_simulations} ({progress_pct:.2f}%)")

        print(f"[INFO] Done. Parquet parts saved to: {parquet_dir}")
        print(f"[INFO] Metadata saved to: {meta_path}")

    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] An error occurred: {e}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()


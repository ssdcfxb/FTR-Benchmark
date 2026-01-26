# -*- coding: utf-8 -*-
"""
====================================
@File Name ：display.py
@Description : Script to spawn multiple robots on various terrains, let them fall and stabilize, then record their states.
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

# add argparse arguments
parser = argparse.ArgumentParser(description="Monte Carlo sampling of robot stable states on various terrains.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--num_rounds", type=int, default=10, help="Number of sampling rounds.")
parser.add_argument("--scan_range", type=float, default=2.0, help="Range (in meters) to scan/shift spawn positions across rounds.")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task.")
parser.add_argument("--terrain", type=str, default="cur_mixed", help="Name of the terrain to use (e.g., cur_mixed, cur_stairs_up).")
parser.add_argument("--spawn_height", type=float, default=0.2, help="Height to spawn robots above the ground.")
parser.add_argument("--robot_spacing", type=float, default=2.0, help="Distance between robots in the grid (meters). Smaller values = tighter spacing.")
parser.add_argument("--settle_steps", type=int, default=50, help="Number of steps to wait for robots to settle.")
parser.add_argument("--sample_steps", type=int, default=20, help="Number of steps to record after settling.")
parser.add_argument("--flipper_front_angle", type=float, default=0.0, help="Target angle (in degrees) for front flipper joints.")
parser.add_argument("--flipper_rear_angle", type=float, default=0.0, help="Target angle (in degrees) for rear flipper joints.")
parser.add_argument("--output_dir", type=str, default="logs/data_collection", help="Directory to save collected data.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Override specific arguments for data collection
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
import ftr_envs.tasks  # Register tasks
from ftr_envs.tasks.crossing.ftr_env import FtrEnv
from omni.isaac.lab.utils.math import euler_xyz_from_quat

def wait_for_user_input(app, prompt="Press Enter to continue..."):
    """
    Waits for user input in the terminal while keeping the Omniverse application updating.
    This allows the user to interact with the viewer (move camera) while the script is paused.
    """
    import select
    print(f"[INFO] {prompt}")
    print("[INFO] Simulation paused. You can move the camera in the viewer.")
    print("[INFO] Press ENTER in the terminal to resume execution.")
    
    while True:
        # Keep the simulator viewer alive/responsive
        app.update()
        
        # Check for input on stdin (non-blocking)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            break

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
    
    print(f"[INFO] Environment created with {args_cli.num_envs} instances on terrain '{args_cli.terrain}'.")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args_cli.output_dir, f"{args_cli.terrain}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    csv_file = os.path.join(save_path, "robot_states.csv")
    static_csv_file = os.path.join(save_path, f"static_{args_cli.terrain}_state.csv")
    first_write = True

    try:
        # Access the underlying environment wrapper (FtrEnv instance)
        unwrapped_env = env.unwrapped
        
        # Access robot directly from the environment instance
        # FtrEnv stores the robot articulation in self._robot
        robot = unwrapped_env._robot

        for round_idx in range(args_cli.num_rounds):
            data_records = []
            
            # --- Offset Calculation ---
            if args_cli.num_rounds > 1:
                progress = round_idx / (args_cli.num_rounds - 1)
                offset_val = -0.5 * args_cli.scan_range + (progress * args_cli.scan_range)
            else:
                offset_val = 0.0

            grid_size = int(np.ceil(np.sqrt(env.num_envs)))
            print(f"\n[INFO] --- Round {round_idx+1}/{args_cli.num_rounds} ---")
            print(f"[INFO] Spawn Offset: {offset_val:.2f} m")
            print(f"[INFO] Robot spacing: {args_cli.robot_spacing} m (Grid: {grid_size}x{grid_size})")

            # Wait for user input (Non-blocking for viewer)
            # wait_for_user_input(simulation_app, f"Waiting for user input to start round {round_idx + 1}...")

            # Reset environment
            env.reset()
            
            # --- State Initialization (Grid Spawn) ---
            # Get root state and apply grid arrangement
            root_state = robot.data.root_state_w.clone() # [N, 13]
            spawn_positions = np.zeros((env.num_envs, 3))

            # --- Set Initial Flipper Positions (Direct Reset) ---
            # Set flipper positions BEFORE setting root state to avoid physics artifacts
            flipper_init_angles = torch.zeros((env.num_envs, 4), device=unwrapped_env.device)
            flipper_init_angles[:, 0] = args_cli.flipper_front_angle
            flipper_init_angles[:, 1] = args_cli.flipper_front_angle
            flipper_init_angles[:, 2] = args_cli.flipper_rear_angle
            flipper_init_angles[:, 3] = args_cli.flipper_rear_angle
            robot.set_all_flipper_positions(flipper_init_angles, degree=True)
            
            for env_id in range(env.num_envs):
                row = env_id // grid_size
                col = env_id % grid_size
                
                grid_x = col * args_cli.robot_spacing
                grid_y = row * args_cli.robot_spacing
                
                world_x = grid_x + offset_val
                world_y = grid_y + offset_val
                
                # Terrain height lookup
                ground_z = 0.0
                if hasattr(unwrapped_env, "terrain_cfg") and hasattr(unwrapped_env.terrain_cfg, "map"):
                    mh = unwrapped_env.terrain_cfg.map
                    search_radius = 1.0
                    px_min = int(np.floor((world_x - search_radius) / mh.cell_size + mh.compensation[0]))
                    px_max = int(np.floor((world_x + search_radius) / mh.cell_size + mh.compensation[0]))
                    py_min = int(np.floor((world_y - search_radius) / mh.cell_size + mh.compensation[1]))
                    py_max = int(np.floor((world_y + search_radius) / mh.cell_size + mh.compensation[1]))
                    
                    x_start = max(0, px_min)
                    x_end = min(mh.map.shape[0], px_max + 1)
                    y_start = max(0, py_min)
                    y_end = min(mh.map.shape[1], py_max + 1)
                    
                    if x_start < x_end and y_start < y_end:
                        ground_z = float(np.max(mh.map[x_start:x_end, y_start:y_end]))
                
                root_state[env_id, 0] = world_x
                root_state[env_id, 1] = world_y
                root_state[env_id, 2] = ground_z + args_cli.spawn_height

                # Set yaw to 180 degrees (Quaternion w,x,y,z = 0,0,0,1)
                root_state[env_id, 3] = 1.0
                root_state[env_id, 4] = 0.0
                root_state[env_id, 5] = 0.0
                root_state[env_id, 6] = 0.0
                
                spawn_positions[env_id, 0] = world_x
                spawn_positions[env_id, 1] = world_y
                spawn_positions[env_id, 2] = root_state[env_id, 2]

                root_state[env_id, 7:10] = 0.0   # Clear linear velocity
                root_state[env_id, 10:13] = 0.0  # Clear angular velocity
            
            # Write state back to sim
            robot.write_root_state_to_sim(root_state, env_ids=torch.arange(env.num_envs, device=unwrapped_env.device))

            # --- Capture Initial States ---
            # root_state indices: 0:3 pos, 3:7 quat (w, x, y, z), 7:10 lin_vel, 10:13 ang_vel
            init_quats = root_state[:, 3:7].clone() # [N, 4]
            # euler_xyz_from_quat returns a tuple (roll, pitch, yaw)
            init_rpy_tuple = euler_xyz_from_quat(init_quats) 
            init_rpy = torch.stack(init_rpy_tuple, dim=1) # [N, 3]
            
            init_quats_np = init_quats.cpu().numpy()
            init_rpy_np = init_rpy.cpu().numpy()

            print(f"[INFO] Simulating for {args_cli.settle_steps} steps to let robots settle...")
            
            # Wait for user input (Non-blocking for viewer)
            # wait_for_user_input(simulation_app, f"Waiting for user input to start round {round_idx + 1}...")
            # Wait for user input
            # print(f"[INFO] Waiting for user input to start")
            # input("Press Enter to continue...")
    
            # Disable resets
            unwrapped_env.suppress_done_signals = True
            
            # --- Set Controls ---
            # Reset velocities commands
            unwrapped_env.forward_vel_commands[:] = 0.0
            
            # Set flipper targets (in radians)
            # FtrEnv uses 4 flippers usually
            unwrapped_env.flipper_target_pos[:] = 0.0 # Reset all
            if unwrapped_env.flipper_target_pos.shape[1] >= 2:
                unwrapped_env.flipper_target_pos[:, 0] = np.deg2rad(args_cli.flipper_front_angle)
                unwrapped_env.flipper_target_pos[:, 1] = np.deg2rad(args_cli.flipper_front_angle)
            if unwrapped_env.flipper_target_pos.shape[1] >= 4:
                unwrapped_env.flipper_target_pos[:, 2] = np.deg2rad(args_cli.flipper_rear_angle)
                unwrapped_env.flipper_target_pos[:, 3] = np.deg2rad(args_cli.flipper_rear_angle)

            # NOTE: We can rely on env.step() to apply these commands via _apply_action()
            # We don't need to manually call robot set methods anymore.

            # Settle Loop
            lin_vel_threshold = 0.1
            ang_vel_threshold = 0.1

            for i in range(args_cli.settle_steps):
                # We need to maintain the action state. FtrEnv doesn't automatically clear them unless reset?
                # _apply_action uses the current values in forward_vel_commands and flipper_target_pos.
                # So we just step.
                
                # We pass zero actions to step(), because pure zero action vector might be interpreted 
                # by the policy wrapper (if any) or _pre_physics_step.
                actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
                
                env.step(actions)

                # if i < 3:

                #     # Wait for user input
                #     print(f"[INFO] Waiting for user input to start round {round_idx + 1}...")
                #     input("Press Enter to continue...")

                
                if (i + 1) % 50 == 0:
                    lin_vel_norm = np.linalg.norm(unwrapped_env.robot_lin_velocities.cpu().numpy(), axis=1)
                    ang_vel_norm = np.linalg.norm(unwrapped_env.robot_ang_velocities.cpu().numpy(), axis=1)
                    is_stable = (lin_vel_norm < lin_vel_threshold) & (ang_vel_norm < ang_vel_threshold)
                    print(f"[INFO] Settle Step {i+1}/{args_cli.settle_steps} - "
                          f"Stable: {int(np.sum(is_stable))}/{env.num_envs}, "
                          f"Mean Vel: {lin_vel_norm.mean():.3f} m/s, "
                          f"Mean AngVel: {ang_vel_norm.mean():.3f} rad/s")

                # Periodically clear joint efforts to ensure passive dynamics if needed
                if i % 10 == 0: 
                    # Although env.step() computes forces, if we want to ensure "floating" drop we might want this,
                    # but usually physics deals with it. The original code did this.
                    # We can use the robot data directly as before if we want to keep this behavior strictly.
                    pass

            print(f"[INFO] Recording data for {args_cli.sample_steps} steps...")
            
            # Sampling Loop
            last_frames_buffer = {env_id: [] for env_id in range(env.num_envs)}
            stable_frame_count = np.zeros(env.num_envs)
            stability_threshold = 20
            
            for i in range(args_cli.sample_steps):
                # Using env properties for clean access
                pos = unwrapped_env.positions.cpu().numpy()
                quat = unwrapped_env.orientations.cpu().numpy()
                euler = unwrapped_env.orientations_3.cpu().numpy()
                lin_vel = unwrapped_env.robot_lin_velocities.cpu().numpy()
                ang_vel = unwrapped_env.robot_ang_velocities.cpu().numpy()
                joint_pos = unwrapped_env.flipper_positions.cpu().numpy()

                lin_vel_norm = np.linalg.norm(lin_vel, axis=1)
                ang_vel_norm = np.linalg.norm(ang_vel, axis=1)
                
                # Logic remains same
                is_currently_stable = (lin_vel_norm < lin_vel_threshold) & (ang_vel_norm < ang_vel_threshold)
                stable_frame_count = np.where(is_currently_stable, stable_frame_count + 1, 0)
                is_stable = stable_frame_count >= stability_threshold
                
                print(f"[INFO] Step {i+1}/{args_cli.sample_steps} - Stable robots: {int(np.sum(is_stable))}/{env.num_envs}")
                
                for env_id in range(env.num_envs):
                     # Record logic
                     is_abnormal = (
                        np.isnan(pos[env_id]).any() or 
                        np.isnan(lin_vel[env_id]).any() or
                        np.isnan(ang_vel[env_id]).any() or
                        lin_vel_norm[env_id] > 10.0 or
                        ang_vel_norm[env_id] > 10.0 or
                        pos[env_id, 2] < -5.0 or
                        pos[env_id, 2] > 5.0
                    )
                     
                     record = {
                        "round": round_idx,
                        "robot_id": env_id,
                        "init_pos_x": float(spawn_positions[env_id, 0]),
                        "init_pos_y": float(spawn_positions[env_id, 1]),
                        "init_pos_z": float(spawn_positions[env_id, 2]),
                        "init_quat_0": float(init_quats_np[env_id, 0]),
                        "init_quat_1": float(init_quats_np[env_id, 1]),
                        "init_quat_2": float(init_quats_np[env_id, 2]),
                        "init_quat_3": float(init_quats_np[env_id, 3]),
                        "init_roll": float(init_rpy_np[env_id, 0]),
                        "init_pitch": float(init_rpy_np[env_id, 1]),
                        "init_yaw": float(init_rpy_np[env_id, 2]),
                        "init_flipper_angle_front": args_cli.flipper_front_angle,
                        "init_flipper_angle_rear": args_cli.flipper_rear_angle,
                        "offset": offset_val,
                        "step": i,
                        "is_stable": int(is_stable[env_id]),
                        "is_abnormal": int(is_abnormal),
                        "lin_vel_norm": lin_vel_norm[env_id],
                        "ang_vel_norm": ang_vel_norm[env_id],
                        "stable_frames": int(stable_frame_count[env_id]),
                        "pos_x": float(pos[env_id, 0]),
                        "pos_y": float(pos[env_id, 1]),
                        "pos_z": float(pos[env_id, 2]),
                        "quat_0": float(quat[env_id, 0]),
                        "quat_1": float(quat[env_id, 1]),
                        "quat_2": float(quat[env_id, 2]),
                        "quat_3": float(quat[env_id, 3]),
                        "roll": float(euler[env_id, 0]),
                        "pitch": float(euler[env_id, 1]),
                        "yaw": float(euler[env_id, 2]),
                        "vel_x": float(lin_vel[env_id, 0]),
                        "vel_y": float(lin_vel[env_id, 1]),
                        "vel_z": float(lin_vel[env_id, 2]),
                        "ang_vel_x": float(ang_vel[env_id, 0]),
                        "ang_vel_y": float(ang_vel[env_id, 1]),
                        "ang_vel_z": float(ang_vel[env_id, 2]),
                    }
                     for j_idx, j_val in enumerate(joint_pos[env_id]):
                        record[f"flipper_{j_idx}"] = float(j_val)
                     last_frames_buffer[env_id].append(record)
                     if len(last_frames_buffer[env_id]) > 5:
                        last_frames_buffer[env_id].pop(0)

                # Step
                actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
                env.step(actions)
            
            # Save data logic
            print(f"[INFO] Extracting last 5 frames from buffer...")
            for env_id in range(env.num_envs):
                frames = last_frames_buffer[env_id]
                num_frames = len(frames)
                for frame_idx, record in enumerate(frames):
                    frame_id = frame_idx + 1 if num_frames <= 5 else frame_idx - (num_frames - 5) + 1
                    record["frame_id"] = frame_id
                    data_records.append(record)
            
            df = pd.DataFrame(data_records)
            df.to_csv(csv_file, mode='a', header=first_write, index=False)
            if 'offset' in df.columns:
                df_static = df.drop(columns=['offset'])
            else:
                df_static = df.copy()
            df_static.to_csv(static_csv_file, mode='a', header=first_write, index=False)
            first_write = False
            
            # Keep suppression on
            unwrapped_env.suppress_done_signals = True
            print(f"[SUCCESS] Round {round_idx+1} complete. Saved {len(df)} records.")

        print(f"[INFO] All rounds completed. Data saved to: {csv_file}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] An error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

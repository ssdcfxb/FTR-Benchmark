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
    # Disable randomization events if possible to keep spawn deterministic initially, 
    # but we want random drops, so maybe keep physics randomization.
]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.lab_tasks.utils import parse_env_cfg
import ftr_envs.tasks  # Register tasks


def main():
    # Parse environment configuration
    # This loads the config registered with the task, and applies CLI overrides from sys.argv
    # It sets up num_envs and other hydra parameters correctly
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu" if args_cli.cpu else "cuda:0",
        num_envs=args_cli.num_envs
    )
    
    # Explicitly override terrain if needed (though sys.argv injection should handle it)
    if hasattr(args_cli, "terrain"):
        env_cfg.terrain_name = args_cli.terrain
    
    # Create the environment with the parsed config
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    print(f"[INFO] Environment created with {args_cli.num_envs} instances on terrain '{args_cli.terrain}'.")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args_cli.output_dir, f"{args_cli.terrain}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Data collection buffer
    csv_file = os.path.join(save_path, "robot_states.csv")
    static_csv_file = os.path.join(save_path, f"static_{args_cli.terrain}_state.csv")
    first_write = True

    try:
        # Access the underlying environment wrapper
        if hasattr(env, 'unwrapped'):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env

        # if hasattr(unwrapped_env, '_apply_action'):
        #     print(f"[INFO] _apply_action method found in environment.")

        # if hasattr(unwrapped_env, 'set_all_flipper_position_targets'):
        #     print(f"[INFO] Environment has set_all_flipper_position_targets method.")
        

        # Note: The environment stores the robot in _robot (private convention) or accessed via scene
        # We'll try to access it via the property usually exposed, but here it seems it's _robot
        if hasattr(unwrapped_env, "robot"):
            robot = unwrapped_env.robot
        elif hasattr(unwrapped_env, "_robot"):
            robot = unwrapped_env._robot
        else:
            # Fallback to looking in scene
            robot = unwrapped_env.scene.articulations["robot"]

        # if hasattr(robot, 'set_all_flipper_position_targets'):
        #     print(f"[INFO] Robot has set_all_flipper_position_targets method.")

        # while True:
        #     a = 1


        # Loop over rounds
        for round_idx in range(args_cli.num_rounds):
            data_records = []
            
            # Calculate offset for this round
            if args_cli.num_rounds > 1:
                # Progress from 0.0 to 1.0
                progress = round_idx / (args_cli.num_rounds - 1)
                # Shift from -half_range to +half_range
                offset_val = -0.5 * args_cli.scan_range + (progress * args_cli.scan_range)
            else:
                offset_val = 0.0

            # Calculate grid size for logging
            grid_size = int(np.ceil(np.sqrt(env.num_envs)))

            print(f"\n[INFO] --- Round {round_idx+1}/{args_cli.num_rounds} ---")
            print(f"[INFO] Spawn Offset: {offset_val:.2f} m")
            print(f"[INFO] Robot spacing: {args_cli.robot_spacing} m (Grid: {grid_size}x{grid_size})")

            # Always reset the environment at the start of each round
            # This ensures robots start from a fresh "initial state" (flat orientation, zero velocity)
            # instead of inheriting the tumbled state from the previous round.
            obs, info = env.reset()
            if round_idx == 0:
                print(f"[INFO] Initial environment reset done.")
            else:
                print(f"[INFO] Environment reset for round {round_idx+1} to restore initial state.")
            
            print(f"[INFO] Spawning robots at height {args_cli.spawn_height}m...")
            print(f"[INFO] Setting flipper target angles - Front: {args_cli.flipper_front_angle:.2f}°, Rear: {args_cli.flipper_rear_angle:.2f}°...")
            
            # Prepare env_ids for operation on all envs
            all_env_ids = torch.arange(env.num_envs, device=unwrapped_env.device)
            
            # Get the root state of the robot and apply offset
            # root_state format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            root_state = robot.data.root_state_w.clone()
            
            # Helper function to print position/velocity/torque statistics
            def print_state_stats(label, pos, vel, torques=None):
                print(f"[DEBUG] Robot positions {label}:")
                print(f"  X: min={pos[:, 0].min():.3f}, max={pos[:, 0].max():.3f}, mean={pos[:, 0].mean():.3f}")
                print(f"  Y: min={pos[:, 1].min():.3f}, max={pos[:, 1].max():.3f}, mean={pos[:, 1].mean():.3f}")
                print(f"  Z: min={pos[:, 2].min():.3f}, max={pos[:, 2].max():.3f}, mean={pos[:, 2].mean():.3f}")
                print(f"[DEBUG] Robot velocities {label}:")
                print(f"  Vx: min={vel[:, 0].min():.3f}, max={vel[:, 0].max():.3f}, mean={vel[:, 0].mean():.3f}")
                print(f"  Vy: min={vel[:, 1].min():.3f}, max={vel[:, 1].max():.3f}, mean={vel[:, 1].mean():.3f}")
                print(f"  Vz: min={vel[:, 2].min():.3f}, max={vel[:, 2].max():.3f}, mean={vel[:, 2].mean():.3f}")
                if torques is not None:
                    torques_abs = np.abs(torques)
                    print(f"[DEBUG] Joint torques {label}:")
                    print(f"  Mean: {torques_abs.mean():.6f}, Std: {torques_abs.std():.6f}")
                    print(f"  Min: {torques_abs.min():.6f}, Max: {torques_abs.max():.6f}")
            
            # DEBUG: Print initial state before modification
            # pos_before = root_state[:, 0:3].cpu().numpy()
            # vel_before = root_state[:, 7:10].cpu().numpy()
            # torques_before = robot.data.joint_torques.cpu().numpy() if hasattr(robot.data, "joint_torques") else None
            
            # Modify root state: apply offsets and clear velocities
            # Create a grid layout for robots with adjustable spacing
            num_envs = env.num_envs
            grid_size = int(np.ceil(np.sqrt(num_envs)))  # e.g., 64 envs -> 8x8 grid
            
            for env_id in range(num_envs):
                row = env_id // grid_size
                col = env_id % grid_size
                
                # Calculate grid position with specified spacing
                grid_x = col * args_cli.robot_spacing
                grid_y = row * args_cli.robot_spacing
                
                # Calculate world position
                world_x = grid_x + offset_val
                world_y = grid_y + offset_val
                
                # Adaptive Terrain Height Lookup
                ground_z = 0.0
                if hasattr(unwrapped_env, "terrain_cfg") and hasattr(unwrapped_env.terrain_cfg, "map"):
                    mh = unwrapped_env.terrain_cfg.map
                    
                    # Search for max height in 4m^2 area (2m x 2m square -> +/- 1.0m radius)
                    search_radius = 1.0
                    
                    # Calculate grid index range
                    px_min = int(np.floor((world_x - search_radius) / mh.cell_size + mh.compensation[0]))
                    px_max = int(np.floor((world_x + search_radius) / mh.cell_size + mh.compensation[0]))
                    py_min = int(np.floor((world_y - search_radius) / mh.cell_size + mh.compensation[1]))
                    py_max = int(np.floor((world_y + search_radius) / mh.cell_size + mh.compensation[1]))
                    
                    # Clamp to map boundaries
                    x_start = max(0, px_min)
                    x_end = min(mh.map.shape[0], px_max + 1)
                    y_start = max(0, py_min)
                    y_end = min(mh.map.shape[1], py_max + 1)
                    
                    # Get max height in the region
                    if x_start < x_end and y_start < y_end:
                        ground_z = float(np.max(mh.map[x_start:x_end, y_start:y_end]))
                
                # Apply offsets
                root_state[env_id, 0] = world_x
                root_state[env_id, 1] = world_y
                # Set Z to terrain height + spawn_height buffer
                # This ensures robots spawn just above the terrain regardless of terrain variation
                root_state[env_id, 2] = ground_z + args_cli.spawn_height
                
                root_state[env_id, 7:10] = 0.0   # Clear linear velocity
                root_state[env_id, 10:13] = 0.0  # Clear angular velocity

            
            # Write root state to simulation
            robot.write_root_state_to_sim(root_state, env_ids=all_env_ids)
            
            # DEBUG: Print state after write
            # pos_after = robot.data.root_state_w[:, 0:3].cpu().numpy()
            # vel_after = robot.data.root_state_w[:, 7:10].cpu().numpy()
            # torques_after = robot.data.joint_torques.cpu().numpy() if hasattr(robot.data, "joint_torques") else None
            
            # Simulation Loop for Settling
            print(f"[INFO] Simulating for {args_cli.settle_steps} steps to let robots settle...")
            
            # Disable automatic resets during settle phase
            if hasattr(unwrapped_env, 'suppress_done_signals'):
                unwrapped_env.suppress_done_signals = True
            
            # **CRITICAL FIX**: Clear control commands from previous round/initialization
            # These commands persist and get applied in every env.step() via _apply_action()
            if hasattr(unwrapped_env, 'forward_vel_commands'):
                unwrapped_env.forward_vel_commands[:] = 0.0
            
            # Set flipper target positions to specified angles
            # Use the robot's set_all_flipper_position_targets method for proper application
            if hasattr(robot, 'set_all_flipper_position_targets'):
                # Create flipper position tensor based on flipper configuration
                # For front and rear flippers, we need to set all 4 flippers
                num_flippers = unwrapped_env.flipper_num
                flipper_angles = torch.zeros((env.num_envs, 4), device=unwrapped_env.device)
                
                # Set front flippers (indices 0, 1) to front angle
                flipper_angles[:, 0:2] = args_cli.flipper_front_angle
                # Set rear flippers (indices 2, 3) to rear angle
                flipper_angles[:, 2:4] = args_cli.flipper_rear_angle
                # print(f"1111111111111111111111111111111111111")
                
                # Apply the flipper positions using the robot method
                robot.set_all_fllipper_position_targets(flipper_angles, degree=True, clip_value=60)
                print(f"[INFO] Flipper targets set: front={args_cli.flipper_front_angle:.2f}°, rear={args_cli.flipper_rear_angle:.2f}°")
                # print(f"1111111111111111111111111111111111111")
            elif hasattr(unwrapped_env, 'flipper_target_pos'):
                # Fallback: set via environment's flipper_target_pos attribute
                # NOTE: flipper_target_pos stores angles in DEGREES (not radians!)
                # Set all 4 flipper joints: 0,1=front left/right, 2,3=rear left/right
                if unwrapped_env.flipper_target_pos.shape[1] >= 2:
                    unwrapped_env.flipper_target_pos[:, 0] = args_cli.flipper_front_angle  # Front left (degrees)
                    unwrapped_env.flipper_target_pos[:, 1] = args_cli.flipper_front_angle  # Front right (degrees)
                if unwrapped_env.flipper_target_pos.shape[1] >= 4:
                    unwrapped_env.flipper_target_pos[:, 2] = args_cli.flipper_rear_angle   # Rear left (degrees)
                    unwrapped_env.flipper_target_pos[:, 3] = args_cli.flipper_rear_angle   # Rear right (degrees)
                print(f"2222222222222222222222222222222222222222222222222")
            
            # while True:
            #     a=1

            # Sample at key points to debug state changes
            sample_indices = [0, args_cli.settle_steps // 4, args_cli.settle_steps // 2, 3 * args_cli.settle_steps // 4, args_cli.settle_steps - 1]
            
            # Stability criteria thresholds
            lin_vel_threshold = 0.1  # m/s
            ang_vel_threshold = 0.1  # rad/s

            for i in range(args_cli.settle_steps):
                # **CRITICAL**: Maintain flipper target position throughout settle phase
                # This must be set BEFORE each step because _apply_action() uses it
                # NOTE: flipper_target_pos expects DEGREES, not radians!
                # Must set ALL 4 flipper joints (front left/right, rear left/right)
                if hasattr(unwrapped_env, 'flipper_target_pos'):
                    if unwrapped_env.flipper_target_pos.shape[1] >= 2:
                        unwrapped_env.flipper_target_pos[:, 0] = args_cli.flipper_front_angle
                        unwrapped_env.flipper_target_pos[:, 1] = args_cli.flipper_front_angle
                    if unwrapped_env.flipper_target_pos.shape[1] >= 4:
                        unwrapped_env.flipper_target_pos[:, 2] = args_cli.flipper_rear_angle
                        unwrapped_env.flipper_target_pos[:, 3] = args_cli.flipper_rear_angle
                
                # Zero action to let gravity act on the robot
                actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
                
                obs, rew, done, truncated, info = env.step(actions)
                
                # Monitor settlement progress every 50 steps
                if (i + 1) % 50 == 0:
                    # Capture State
                    lin_vel = unwrapped_env.robot_lin_velocities.cpu().numpy() # [N, 3]
                    ang_vel = unwrapped_env.robot_ang_velocities.cpu().numpy() # [N, 3]
                    
                    # Check stability
                    lin_vel_norm = np.linalg.norm(lin_vel, axis=1)
                    ang_vel_norm = np.linalg.norm(ang_vel, axis=1)
                    is_stable = (lin_vel_norm < lin_vel_threshold) & (ang_vel_norm < ang_vel_threshold)
                    
                    print(f"[INFO] Settle Step {i+1}/{args_cli.settle_steps} - "
                          f"Stable: {int(np.sum(is_stable))}/{env.num_envs}, "
                          f"Mean Vel: {lin_vel_norm.mean():.3f} m/s, "
                          f"Mean AngVel: {ang_vel_norm.mean():.3f} rad/s")

                # Periodically clear joint efforts
                if i % 10 == 0 and hasattr(robot, "data"):
                    if hasattr(robot.data, "joint_torques"):
                        robot.data.joint_torques[:] = 0.0
                    if hasattr(robot.data, "joint_forces"):
                        robot.data.joint_forces[:] = 0.0
            
            print(f"[INFO] Recording data for {args_cli.sample_steps} steps...")
            
            # **CRITICAL FIX**: Keep suppress_done_signals=True during sampling phase
            # This prevents robots from being reset during data collection
            if hasattr(unwrapped_env, 'suppress_done_signals'):
                unwrapped_env.suppress_done_signals = True
                print(f"[INFO] Keeping suppress_done_signals enabled during sampling phase")
            
            # Stability tracking: count consecutive stable frames per environment
            stable_frame_count = np.zeros(env.num_envs)
            stability_threshold = 20  # Need 20 consecutive stable frames
            
            # Stability criteria thresholds (Defined above before settle loop)
            # lin_vel_threshold = 0.1  # m/s
            # ang_vel_threshold = 0.1  # rad/s
            
            # Buffer to store the last 5 frames of data for each environment
            last_frames_buffer = {env_id: [] for env_id in range(env.num_envs)}
            
            # Data Collection Loop
            for i in range(args_cli.sample_steps):
                # 1. Capture State
                # Positions (x, y, z)
                pos = unwrapped_env.positions.cpu().numpy() # [N, 3]
                
                # Orientations (Quat: w, x, y, z or x, y, z, w?)
                # unwrapped_env.orientations usually provides standardized quats.
                quat = unwrapped_env.orientations.cpu().numpy() # [N, 4]
                
                # Euler Angles (Roll, Pitch, Yaw)
                # unwrapped_env.orientations_3 is computed in FtrEnv usually
                euler = unwrapped_env.orientations_3.cpu().numpy() # [N, 3]
                
                # Linear Velocity
                lin_vel = unwrapped_env.robot_lin_velocities.cpu().numpy() # [N, 3]
                
                # Angular Velocity
                ang_vel = unwrapped_env.robot_ang_velocities.cpu().numpy() # [N, 3]
                
                # Joint Positions (Flippers)
                joint_pos = unwrapped_env.flipper_positions.cpu().numpy() # [N, num_flippers]
                
                # 2. Check stability using multiple criteria
                lin_vel_norm = np.linalg.norm(lin_vel, axis=1)
                ang_vel_norm = np.linalg.norm(ang_vel, axis=1)
                
                # Stable if both linear and angular velocities are below thresholds
                is_currently_stable = (lin_vel_norm < lin_vel_threshold) & (ang_vel_norm < ang_vel_threshold)
                
                # Update stability frame counter
                stable_frame_count = np.where(is_currently_stable, stable_frame_count + 1, 0)
                
                # Only record data for robots that have been stable for enough frames
                is_stable = stable_frame_count >= stability_threshold
                
                print(f"[INFO] Step {i+1}/{args_cli.sample_steps} - Stable robots: {int(np.sum(is_stable))}/{env.num_envs}")
                
                # DEBUG: Print more details (Commented out for clean output)
                # if i < 3 or i == args_cli.sample_steps - 1:
                #     print(f"[DEBUG] Sample step {i}: Z_pos mean={pos.mean(axis=0)[2]:.3f}, lin_vel_norm mean={lin_vel_norm.mean():.3f}, ang_vel_norm mean={ang_vel_norm.mean():.3f}")
                #     if hasattr(robot, "data") and hasattr(robot.data, "joint_torques"):
                #         torques = robot.data.joint_torques.cpu().numpy()
                #         print(f"[DEBUG] Joint torques: mean={np.abs(torques).mean():.6f}, max={np.abs(torques).max():.6f}")
                
                # 3. Store Data for all robots, track stability status
                for env_id in range(env.num_envs):
                    # Determine if robot is abnormal (extreme values)
                    is_abnormal = (
                        np.isnan(pos[env_id]).any() or 
                        np.isnan(lin_vel[env_id]).any() or
                        np.isnan(ang_vel[env_id]).any() or
                        lin_vel_norm[env_id] > 10.0 or  # Linear velocity too high
                        ang_vel_norm[env_id] > 10.0 or  # Angular velocity too high
                        pos[env_id, 2] < -5.0 or  # Z position too low (fallen far)
                        pos[env_id, 2] > 5.0  # Z position too high (flying)
                    )
                    
                    record = {
                        "round": round_idx,
                        "robot_id": env_id,
                        "offset": offset_val,
                        "step": i,
                        "flipper_angle_front": args_cli.flipper_front_angle,
                        "flipper_angle_rear": args_cli.flipper_rear_angle,
                        "is_stable": int(is_stable[env_id]),  # 1 if stable, 0 if not
                        "is_abnormal": int(is_abnormal),  # 1 if abnormal, 0 if normal
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
                        # Flatten joint positions
                    }
                    # Add joint positions dynamically (flipper angles)
                    for j_idx, j_val in enumerate(joint_pos[env_id]):
                        record[f"flipper_{j_idx}"] = float(j_val)
                    
                    # Store frame in buffer (keep only last 5 frames) for all robots
                    last_frames_buffer[env_id].append(record)
                    if len(last_frames_buffer[env_id]) > 5:
                        last_frames_buffer[env_id].pop(0)

                # **CRITICAL**: Maintain flipper target position throughout sampling phase
                # This must be set BEFORE each step because _apply_action() uses it
                # NOTE: flipper_target_pos expects DEGREES, not radians!
                # Must set ALL 4 flipper joints (front left/right, rear left/right)
                if hasattr(unwrapped_env, 'flipper_target_pos'):
                    if unwrapped_env.flipper_target_pos.shape[1] >= 2:
                        unwrapped_env.flipper_target_pos[:, 0] = args_cli.flipper_front_angle
                        unwrapped_env.flipper_target_pos[:, 1] = args_cli.flipper_front_angle
                    if unwrapped_env.flipper_target_pos.shape[1] >= 4:
                        unwrapped_env.flipper_target_pos[:, 2] = args_cli.flipper_rear_angle
                        unwrapped_env.flipper_target_pos[:, 3] = args_cli.flipper_rear_angle
                
                # Step simulation to keep physics running
                # Zero action and ensure no joint torques
                actions = torch.zeros((env.num_envs, unwrapped_env.action_space.shape[1]), device=unwrapped_env.device)
                obs, rew, done, truncated, info = env.step(actions)
                
                # **CRITICAL DIAGNOSTIC**: Track done signals during sampling
                # done_numpy = done.cpu().numpy() if isinstance(done, torch.Tensor) else done
                # if np.any(done_numpy):
                #     reset_count = np.sum(done_numpy)
                #     reset_indices = np.where(done_numpy)[0]
                #     print(f"\n[ALERT] Sample step {i}: {reset_count} environments triggered done signal!")
                #     print(f"  Reset env_ids: {reset_indices[:5]}{'...' if len(reset_indices) > 5 else ''}")
                #     print(f"  forward_vel_commands after reset: {unwrapped_env.forward_vel_commands[reset_indices[:3]].cpu().numpy()}")
                    
                #     # Check positions - they should have changed dramatically if reset
                #     pos_after = robot.data.root_state_w[reset_indices, 0:3].cpu().numpy()
                #     vel_after = robot.data.root_state_w[reset_indices, 7:10].cpu().numpy()
                #     print(f"  New positions: {pos_after[:3]}")
                #     print(f"  New velocities: {vel_after[:3]}")
                #     print()
                
                # DEBUG: Check if done signal triggered
                # done_numpy_check = done.cpu().numpy() if isinstance(done, torch.Tensor) else done
                # if np.any(done_numpy_check) and i < 3:
                #     print(f"[WARNING] Sample step {i}: {np.sum(done_numpy_check)} environments triggered done signal! Robot positions after step:")
                #     pos_after = robot.data.root_state_w[:, 2].cpu().numpy()
                #     print(f"[DEBUG] Z_pos after done: mean={pos_after.mean():.3f}, range=[{pos_after.min():.3f}, {pos_after.max():.3f}]")
                
                # Periodically clear joint efforts to ensure passive dynamics
                if i % 10 == 0 and hasattr(robot, "data"):
                    if hasattr(robot.data, "joint_torques"):
                        robot.data.joint_torques[:] = 0.0
                    if hasattr(robot.data, "joint_forces"):
                        robot.data.joint_forces[:] = 0.0

            # Extract last 5 frames from buffer and add frame_id (1-5)
            print(f"[INFO] Extracting last 5 frames from buffer...")
            for env_id in range(env.num_envs):
                frames = last_frames_buffer[env_id]
                num_frames = len(frames)
                for frame_idx, record in enumerate(frames):
                    # Calculate frame_id from 1-5 based on position in the frames list
                    frame_id = frame_idx + 1 if num_frames <= 5 else frame_idx - (num_frames - 5) + 1
                    record["frame_id"] = frame_id
                    data_records.append(record)
            
            # Save to CSV after every round to prevent data loss
            df = pd.DataFrame(data_records)
            df.to_csv(csv_file, mode='a', header=first_write, index=False)
            
            # Save second copy without 'offset' column
            if 'offset' in df.columns:
                df_static = df.drop(columns=['offset'])
            else:
                df_static = df.copy()
            df_static.to_csv(static_csv_file, mode='a', header=first_write, index=False)
            
            first_write = False
            print(f"[SUCCESS] Round {round_idx+1} complete. Saved {len(df)} records.")
            
            # **CRITICAL FIX**: Keep suppress_done_signals=True between rounds
            # This ensures robots aren't reset when transitioning to the next round's settle phase
            if hasattr(unwrapped_env, 'suppress_done_signals'):
                unwrapped_env.suppress_done_signals = True
                print(f"[INFO] Keeping suppress_done_signals enabled for next round")
            
        print(f"[INFO] All rounds completed. Data saved to: {csv_file}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] An error occurred: {e}")
    finally:
        # close the environment
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

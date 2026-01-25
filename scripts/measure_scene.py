# -*- coding: utf-8 -*-
"""
====================================
@File Name ：measure_scene.py
@Description : Script to load the environment (Robot + Terrain) and keep it running for manual measurement using Isaac Sim GUI tools.
====================================
"""

import argparse
import sys
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Load environment for manual measurement.")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task.")
parser.add_argument("--terrain", type=str, default="cur_mixed", help="Name of the terrain to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Override specific arguments to force GUI mode if not set (though launch app handles it)
sys.argv = [sys.argv[0]] + hydra_args + [
    f"env.scene.num_envs={args_cli.num_envs}",
    f"env.terrain_name={args_cli.terrain}",
]

# Launch Omniverse Application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports must happen after app launch"""
import gymnasium as gym
import torch
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
import ftr_envs.tasks  # Register custom tasks

def main():
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu", # Use CPU for simple interaction, change to cuda:0 if physics is heavy
        num_envs=args_cli.num_envs
    )
    
    if hasattr(args_cli, "terrain"):
        env_cfg.terrain_name = args_cli.terrain

    # Create the environment
    print(f"[INFO] Creating environment with terrain '{args_cli.terrain}'...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # Check for unwrapped env to get device
    if hasattr(env, 'unwrapped'):
        unwrapped_env = env.unwrapped
        device = unwrapped_env.device
        num_actions = unwrapped_env.action_space.shape[1]
    else:
        device = "cpu"
        num_actions = env.action_space.shape[1]

    # Initial Reset
    print("[INFO] Resetting environment...")
    env.reset()

    # Instructions for the user
    print("\n" + "=" * 80)
    print(" SCENE LOADED SUCCESSFULLY")
    print("=" * 80)
    print(">>> HOW TO PAUSE SIMULATION:")
    print("   Method 1: Press 'Spacebar' on your keyboard.")
    print("   Method 2: Click the 'Pause' button (||) on the left side of the Viewport toolbar.")
    print("-" * 80)
    print(">>> HOW TO FIND THE MEASURE TOOL:")
    print("   Method 1 (Best): In the top menu bar, go to 'Tools' -> 'Measure Tool'.")
    print("   Method 2: Look for the 'Ruler' icon in the vertical toolbar on the RIGHT side.")
    print("-" * 80)
    print(">>> HOW TO MEASURE:")
    print("   1. Activate the tool (Popup window 'Measure Tool' appears).")
    print("   2. (Recommended) Click the MAGNET icon in that popup to enable 'Vertex Snap'.")
    print("   3. Click Start Point -> Click End Point.")
    print("=" * 80)
    print("Press 'Ctrl+C' in this terminal to close the application.")
    print("=" * 80 + "\n")

    try:
        # Freeze the robot after initial settling if needed, or just let it rest.
        # If the robot is sliding, we can force reset its velocity every frame or disable physics stepping.
        
        while simulation_app.is_running():
            # Step the simulation with zero actions to keep the physics running but robot static
            # This allows you to inspect the robot in a 'live' state (gravity applies)
            actions = torch.zeros((args_cli.num_envs, num_actions), device=device)
            
            # Step physics
            env.step(actions)
            
            # [Optional] Force stop robot if it's sliding on slopes
            # if hasattr(unwrapped_env, "_robot"):
            #     unwrapped_env._robot.write_root_velocity_to_sim(torch.zeros((args_cli.num_envs, 6), device=device))

            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt received. Closing...")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()

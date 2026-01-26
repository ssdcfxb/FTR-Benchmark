# -*- coding: utf-8 -*-
"""
====================================
@File Name : empty_scene.py
@Description : Script to launch the simulator and create an empty scene in Isaac Lab.
====================================
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# Create the argument parser
parser = argparse.ArgumentParser(description="Launch an empty Isaac Lab scene.")
# Add AppLauncher arguments (e.g. --headless, --gpu_id)
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args_cli = parser.parse_args()

# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports must happen AFTER the app is launched
from omni.isaac.lab.sim import SimulationContext, SimulationCfg

def main():
    """Main function to run the simulation."""
    
    # Configure the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    sim = SimulationContext(sim_cfg)
    
    # Configure Camera settings
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Import necessary modules for USD Stage and Prim management
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    
    # Path to the USD file
    usd_path = "/home/ps/FTR-Benchmark/ftr_envs/assets/usd/ftr/ftr_v1.usd"
    
    # Add reference to stage
    # prim_path is where the robot will be mounted in the USD hierarchy
    prim_utils.create_prim(
        prim_path="/World/Robot",
        usd_path=usd_path,
        translation=[0.0, 0.0, 1.0] # Spawn higher (1.0m) to avoid floor clipping
    )

    # ADD A LIGHT
    import omni.kit.commands
    from pxr import UsdLux, Sdf
    stage = omni.usd.get_context().get_stage()
    light_prim = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/defaultDomeLight"))
    light_prim.CreateIntensityAttr(1000)

    # COMPULSORY GROUND PLANE (Physics based)
    # Don't rely on network assets strictly, create a local physics plane
    from omni.isaac.core.objects import FixedCuboid
    import numpy as np
    
    # Create a giant cube as the ground
    # Size 1.0, Scale 200 -> 200m x 200m
    # Z-scale 1.0 -> 1m thickness.
    # Position Z = -0.5 -> Top surface is at 0.0
    FixedCuboid(
        prim_path="/World/GroundPlane",
        position=np.array([0, 0, -0.5]),
        scale=np.array([200.0, 200.0, 1.0]),
        size=1.0,
        color=np.array([0.3, 0.3, 0.3])
    )
    
    # sim.reset() is needed to register the new physics object
    # sim.reset() # We will reset later after fixing masses

    # ---------------------------------------------------------
    # HOTFIX: Fix invalid mass properties that cause falling/flying
    # ---------------------------------------------------------
    from pxr import UsdPhysics, Usd
    print("[INFO] Checking for invalid mass properties...")
    stage = omni.usd.get_context().get_stage()
    
    count_fixed = 0
    
    for prim in stage.Traverse():
        # Check if it is a Rigid Body (Physics Body)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            current_mass = None
            
            # Check explicit MassAPI first
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                current_mass = mass_api.GetMassAttr().Get()
            
            # Fallback: Check for legacy physics:mass attribute
            if current_mass is None:
                attr = prim.GetAttribute("physics:mass")
                if attr and attr.IsValid():
                    current_mass = attr.Get()

            # If mass is missing or invalid (<=0), force a fix
            if current_mass is None or current_mass <= 1e-6:
                print(f"  [FIX] RigidBody {prim.GetPath()} has invalid mass ({current_mass}). Forcing MassAPI mass=1.0")
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.GetMassAttr().Set(1.0)
                count_fixed += 1

    if count_fixed > 0:
        print(f"[INFO] Fixed {count_fixed} prims with invalid/missing mass.")
    else:
        print("[INFO] No invalid masses found via traversal.")

    sim.reset()
    
    print("-" * 80)
    print(f"[INFO] Loaded USD: {usd_path}")
    print("[INFO] Simulation is running.")
    print("-" * 80)

    while simulation_app.is_running():
        sim.step()

    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        # Close the simulator
        simulation_app.close()

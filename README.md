# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import os
import carb

from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader

class FrankaKinematicsExample():
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None

        self._waypoints = [
            (np.array([0.3,  0.0,  0.5]), euler_angles_to_quats([0, np.pi, 0])),  # Start
            (np.array([0.2,  0.0,  0.5]), euler_angles_to_quats([0, np.pi, 0])),  # Left 1
            (np.array([0.1,  0.0,  0.5]), euler_angles_to_quats([0, np.pi, 0])),  # Left 2
            (np.array([0.1,  0.0,  0.4]), euler_angles_to_quats([0, np.pi, 0])),  # Down 1
            (np.array([0.1,  0.0,  0.3]), euler_angles_to_quats([0, np.pi, 0]))   # Down 2
        ]
        self._current_wp = 0
        self._time_since_last_wp = 0.0
        self._wp_delay = 5.0  # seconds delay between moves

    def load_example_assets(self):
        # Add the Franka and target to the stage

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)
        
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])
        self._target.set_default_state(np.array([.3,0,.5]),euler_angles_to_quats([0,np.pi,0]))

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "right_gripper"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)


    def update(self, step: float):
        self._time_since_last_wp += step

        if self._time_since_last_wp >= self._wp_delay:
            self._current_wp = (self._current_wp + 1) % len(self._waypoints)
            self._time_since_last_wp = 0.0

        target_position, target_orientation = self._waypoints[self._current_wp]

        # Optional: update visual marker if you're using one
        if self._target:
            self._target.set_world_pose(target_position, target_orientation)

        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution.")


    def reset(self):
        # Kinematics is stateless
        pass


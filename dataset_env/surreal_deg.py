import os
import sys
import numpy as np
from collections import OrderedDict

from deg_base import DataEnvGroup
from file_storage import store_trajectoy

ENV_PATH = '/scratch/envs/robosuite'
sys.path.append(ENV_PATH)

import robosuite as suite
from robosuite.wrappers import IKWrapper
import robosuite.utils.transform_utils as T

class SurrealDataEnvGroup(DataEnvGroup):
    ''' DataEnvGroup for Surreal Robotics Suite environment. 
        
        + The observation space can be modified through `global_config.env_args`
        + Observation space:
            - 'robot-state': proprioceptive feature - vector of:
                > cos and sin of robot joint positions
                > robot joint velocities 
                > current configuration of the gripper.
            - 'object-state': object-centric feature 
            - 'image': RGB/RGB-D image 
                > (256 x 256 by default)
            - Refer: https://github.com/StanfordVL/robosuite/tree/master/robosuite/environments

        + The action spaces by default are joint velocities and gripper actuations.
            - Dimension : [8]
            - To use the end-effector action-space use inverse-kinematics using IKWrapper.
            - Refer: https://github.com/StanfordVL/robosuite/tree/master/robosuite/wrappers
    '''
    def __init__(self, get_episode_type=None):
        super(SurrealDataEnvGroup, self).__init__(get_episode_type)
        assert self.env_name == 'SURREAL'

        self.vis_obv_key = 'image'
        self.dof_obv_key = 'robot-state'
        self.obs_space = {self.vis_obv_key: (256, 256, 3), self.dof_obv_key: (30)}
        self.action_space = (8)

    def get_env(self):
        env = suite.make(self.config.env_type, **self.config.env_args)
        return env

     def _get_obs(self, obs, key):
        return obs[key]

    def play_trajectory(self):
        # TODO 
        # Refer https://github.com/StanfordVL/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py
        raise NotImplementedError

    def teleoperate(self, demons_config):
        env = self.get_env()
        # Need to use inverse-kinematics controller to set position using device 
        env = IKWrapper(env)
        
        if demons_config.device == "keyboard":
            from robosuite.devices import Keyboard
            device = Keyboard()
            env.viewer.add_keypress_callback("any", device.on_press)
            env.viewer.add_keyup_callback("any", device.on_release)
            env.viewer.add_keyrepeat_callback("any", device.on_press)
        elif demons_config.device == "spacemouse":
            from robosuite.devices import SpaceMouse
            device = SpaceMouse()
        
        for run in range(demons_config.n_runs):
            obs = env.reset()
            env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) 
            # rotate the gripper so we can see it easily - NOTE : REMOVE MAYBE
            env.viewer.set_camera(camera_id=2)
            env.render()
            device.start_control()

            reset = False
            task_completion_hold_count = -1
            step = 0
            tr_vobvs, tr_dof, tr_actions = [], [], []
            
            while not reset:
                if int(step % demons_config.collect_freq) == 0:
                    tr_vobvs.append(np.array(obs[self.vis_obv_key]))
                    tr_dof.append(np.array(obs[self.dof_obv_key].flatten()))
                
                device_state = device.get_controller_state()
                dpos, rotation, grasp, reset = (
                    device_state["dpos"],
                    device_state["rotation"],
                    device_state["grasp"],
                    device_state["reset"],
                )

                current = env._right_hand_orn
                drotation = current.T.dot(rotation)  
                dquat = T.mat2quat(drotation)
                grasp = grasp - 1. 
                ik_action = np.concatenate([dpos, dquat, [grasp]])

                obs, _, done, _ = env.step(ik_action)
                env.render()

                joint_velocities = np.array(env.controller.commanded_joint_velocities)
                if env.env.mujoco_robot.name == "sawyer":
                    gripper_actuation = np.array(ik_action[7:])
                elif env.env.mujoco_robot.name == "baxter":
                    gripper_actuation = np.array(ik_action[14:])

                # NOTE: Action for the normal environment (not inverse kinematic)
                action = np.concatenate([joint_velocities, gripper_actuation], axis=0)
                
                if int(step % demons_config.collect_freq) == 0:
                    tr_actions.append(action)

                if (int(step % demons_config.flush_freq) == 0) or (demons_config.break_traj_success and task_completion_hold_count == 0):
                    print("Storing Trajectory")
                    trajectory = {self.vis_obv_key : np.array(tr_vobvs), self.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
                    store_trajectoy(trajectory, 'teleop')
                    trajectory, tr_vobvs, tr_dof, tr_actions = {}, [], [], []

                if demons_config.break_traj_success and env._check_success():
                    if task_completion_hold_count > 0:
                        task_completion_hold_count -= 1 # latched state, decrement count
                    else:
                        task_completion_hold_count = 10 # reset count on first success timestep
                else:
                    task_completion_hold_count = -1

                step += 1

            env.close()

    def random_trajectory(self, demons_config):
        env = self.get_env()
        obs = env.reset()
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) 

        tr_vobvs, tr_dof, tr_actions = [], [], []
        for step in range(demons_config.flush_freq):
            tr_vobvs.append(np.array(obs[self.vis_obv_key]))
            tr_dof.append(np.array(obs[self.dof_obv_key].flatten()))
            
            action = np.random.randn(env.dof)
            obs, reward, done, info = env.step(action)

            tr_actions.append(action)
        
        print("Storing Trajectory")
        trajectory = {self.vis_obv_key : np.array(tr_vobvs), self.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
        store_trajectoy(trajectory, 'random')
        env.close()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    print("=> Testing surreal_deg.py")

    deg = SurrealDataEnvGroup()
    print(deg.obs_space[deg.vis_obv_key], deg.action_space)
    print(deg.get_env().reset()[deg.vis_obv_key].shape)

    traj_data = DataLoader(deg.traj_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(next(iter(traj_data))[deg.dof_obv_key].shape)

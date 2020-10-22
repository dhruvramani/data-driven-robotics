import os
import sys
import gym
import numpy as np

from deg_base import DataEnvGroup
from file_storage import store_trajectoy

ENV_PATH = '/scratch/envs/RLBench'
sys.path.append(ENV_PATH)

import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

class RLBenchDataEnvGroup(DataEnvGroup):
    ''' DataEnvGroup for RLBench environment. 
        
        + The observation space can be modified through `global_config.env_args`
        + Observation space:
            - 'state': proprioceptive feature : [37] + task_specific
                > robot joint - velocities, positions, forces
                > gripper - pose, joint-position, touch_forces
                > task_low_dim_state
            - RGB-D/RGB image : (128 x 128 by default)
                > 'left_shoulder_rgb', 'right_shoulder_rgb'
                > 'front_rgb', 'wrist_rgb'
            - NOTE : Better to use only one of the RGB obvs rather than all, saves a lot of time while env creation.

            - Refer: https://github.com/stepjam/RLBench/blob/20988254b773aae433146fff3624d8bcb9ed7330/rlbench/observation_config.py

        + The action spaces by default are joint velocities [7] and gripper actuations [1].
            - Dimension : [8]
    '''
    def __init__(self, get_episode_type=None):
        super(RLBenchDataEnvGroup, self).__init__(get_episode_type)
        assert self.env_name == 'RLBENCH'
        self.env_obj = None
        self.use_gym = self.config.env_args['use_gym']

        self.observation_mode = self.config.env_args['observation_mode'].lower()
        self.left_obv_key  = 'left_shoulder_rgb'
        self.right_obv_key = 'right_shoulder_rgb'
        self.wrist_obv_key = 'wrist_rgb'
        self.front_obv_key = 'front_rgb'

        if self.observation_mode not in ['vision', 'state'] and not self.use_gym:
            self.config.env_args['vis_obv_key'] = self.observation_mode

        self.vis_obv_key = self.config.env_args['vis_obv_key']
        self.dof_obv_key = 'state'
        self.env_action_key = 'joint_velocities'
        self.env_gripper_key = 'gripper_open'

        self.obs_space = {self.dof_obv_key : (37), self.left_obv_key : (128, 128, 3), self.right_obv_key : (128, 128, 3),
                    self.wrist_obv_key : (128, 128, 3), self.front_obv_key: (128, 128, 3)}
        
        self.action_space, self.gripper_space = (7), (1)
        if self.config.env_args['combine_action_space']:
            self.action_space += self.gripper_space

    def get_env(self, task=None):
        task = task if task else self.config.env_type
        if self.use_gym:
            assert type(task) == str # NOTE : When using gym, the task has to be represented as a sting.
            assert self.observation_mode in ['vision', 'state']

            env = gym.make(task, observation_mode=self.config.env_args['observation_mode'],
                render_mode=self.config.env_args['render_mode'])
            self.env_obj = env
        else:            
            obs_config = ObservationConfig()
            if self.observation_mode == 'vision':
                obs_config.set_all(True)
            elif self.observation_mode == 'state':
                obs_config.set_all_high_dim(False)
                obs_config.set_all_low_dim(True)
            else:
                obs_config_dict = {self.left_obv_key : obs_config.left_shoulder_camera,
                                self.right_obv_key : obs_config.right_shoulder_camera, 
                                self.wrist_obv_key : obs_config.wrist_camera, 
                                self.front_obv_key: obs_config.front_camera}  

                assert self.observation_mode in obs_config_dict.keys()

                obs_config.set_all_high_dim(False)
                obs_config_dict[self.observation_mode].set_all(True)
                obs_config.set_all_low_dim(True)

            # TODO : Write code to change it from env_args
            action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
            self.env_obj = Environment(action_mode, obs_config=obs_config, headless=True)

            task = task if task else ReachTarget
            if type(task) == str:
                task = task.split('-')[0]
                task = self.env_obj._string_to_task(task)
            
            self.env_obj.launch()
            env = self.env_obj.get_task(task) # NOTE : `env` refered as `task` in RLBench docs. 
        return env

    def _get_obs(self, obs, key):
        assert obs is not None and key is not None
        if type(obs) == tuple:
            obs = obs[1]
        if type(obs) == dict:
            return obs[key]
        elif type(obs) == Observation:
            if key == 'state':
                return obs.get_low_dim_data()
            return getattr(obs, key)

    def shutdown_env(self):
        if self.env_obj is None:
            print("Environment not created, call `.get_env()`")
        elif self.use_gym:
            self.env_obj.close()
        else:
            self.env_obj.shutdown()
        self.env_obj = None

    def teleoperate(self, demons_config, task=None):
        if self.config.env_args['keyboard_teleop']:
            raise NotImplementedError
        else:
            if self.env_obj is None or task is None:
                env = self.get_env(task)
            else:
                if type(task) == str:
                    task = self.env_obj._string_to_task(task.split('-')[0])        
                env = self.env_obj.get_task(task)

            if self.use_gym:
                demos = env.task.get_demos(demons_config.n_runs, live_demos=True)
            else : 
                demos = env.get_demos(demons_config.n_runs, live_demos=True)
            demos = np.array(demos).flatten()

            for i in range(demons_config.n_runs):
                sample = demos[i]
                if self.observation_mode != 'state':
                    tr_vobvs = np.array([self._get_obs(obs, self.vis_obv_key) for obs in sample])
                tr_dof = np.array([self._get_obs(obs, self.dof_obv_key).flatten() for obs in sample])
                tr_actions = np.array([self._get_obs(obs, self.env_action_key).flatten() for obs in sample])
                tr_gripper = np.array([[self._get_obs(obs, self.env_gripper_key)] for obs in sample])

                if self.config.env_args['combine_action_space']:
                    tr_actions = np.concatenate((tr_actions, tr_gripper), axis=-1)

                print("Storing Trajectory")
                trajectory = {self.dof_obv_key : tr_dof, 'action' : tr_actions}
                if self.observation_mode != 'state':
                    trajectory.update({self.vis_obv_key : tr_vobvs})
                store_trajectoy(trajectory, episode_type='teleop', task=task)
            
            self.shutdown_env()

    def random_trajectory(self, demons_config):
        env = self.get_env()
        obs = env.reset()

        tr_vobvs, tr_dof, tr_actions = [], [], []
        for step in range(demons_config.flush_freq):
            if self.observation_mode != 'state':
                tr_vobvs.append(np.array(self._get_obs(obs, self.vis_obv_key)))
            tr_dof.append(np.array(self._get_obs(obs, self.dof_obv_key).flatten()))
            
            action = np.random.normal(size=self.action_space[0])
            obs, reward, done, info = env.step(action)

            tr_actions.append(action)
        
        print("Storing Trajectory")
        trajectory = {self.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
        if self.observation_mode != 'state':
            trajectory.update({self.vis_obv_key : np.array(tr_vobvs)})
        store_trajectoy(trajectory, episode_type='random')
        self.shutdown_env()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    print("=> Testing rlbench_deg.py")

    deg = RLBenchDataEnvGroup()
    print(deg.obs_space[deg.vis_obv_key], deg.action_space)
    env = deg.get_env(task='change_clock')
    obs = env.reset()
    print(deg._get_obs(obs, deg.dof_obv_key).shape)
    print(deg._get_obs(obs, 'task_low_dim_state').shape)

    #traj_data = DataLoader(deg.traj_dataset, batch_size=1, shuffle=False, num_workers=1)
    traj_data = deg.get_traj_dataloader(batch_size=5)
    for i, b in enumerate(traj_data):
       print(i, b[deg.dof_obv_key].shape)

    deg.shutdown_env()

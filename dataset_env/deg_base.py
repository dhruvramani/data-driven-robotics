import os
import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import data_aug as rad
from data_config import get_dataset_args
from file_storage import get_trajectory, get_random_trajectory

class DataEnvGroup(object):
    ''' + NOTE : Create subclass for every environment, eg.
        Check `assert self.env_name == 'ENV_NAME'`
    '''
    def __init__(self, get_episode_type=None):
        ''' + Arguments:
                - get_episode_type: Get data of a particular episode_type (teleop, imitation, etc.)
                    > Default : None, get data with any episode_type.
        '''
        self.config = get_dataset_args()
        self.env_name = self.config.env
        self.env_type = self.config.env_type
        self.episode_type = get_episode_type
        self.traj_dataset = self.TrajDataset(self.episode_type, self.config)

        # NOTE : Environment dependent properties
        # Set these after inheriting the class. NotImplementedError
        self.vis_obv_key = None
        self.dof_obv_key = None
        self.word_embeddings_key = 'word_embeddings'
        self.sentence_embedding_key = 'sentence_embedding'

        self.obs_space = None
        self.action_space = None

    def get_env(self):
        raise NotImplementedError

    def _get_obs(self, obs, key):
        raise NotImplementedError

    def teleoperate(self, demon_config, task=None):
        raise NotImplementedError

    def random_trajectory(self, demons_config):
        raise NotImplementedError

    def get_random_goal(self):
        assert issubclass(type(self), DataEnvGroup) is True # NOTE : might raise error - remove if so
        goal = get_random_trajectory()[0][self.vis_obv_key][-1]
        return goal

    class TrajDataset(Dataset):
        def __init__(self, episode_type, config):
            self.episode_type = episode_type
            self.config = config

        def __len__(self):
            if self.episode_type is None:
                return self.config.traj_db.objects.count() - 1 # HACK
            else:
                return self.config.traj_db.objects.filter(episode_type=self.episode_type).count()

        def __getitem__(self, idx):
            trajectory =  get_trajectory(index=idx, episode_type=self.episode_type)
            if self.config.data_agumentation:
                trajectory[self.vis_obv_key] = rad.apply_augs(trajectory[self.vis_obv_key], self.config)
            return trajectory

    def _collate_wrap(self, remove_task_state=False):
        # NOTE - if batch_size > 1, it removes the task-dependent states to get a common dim.
        def pad_collate(batch):
            assert None not in [self.vis_obv_key, self.dof_obv_key]
            tr_vobvs = [torch.from_numpy(b[self.vis_obv_key]) for b in batch]
            tr_dof = [torch.from_numpy(b[self.dof_obv_key][:, : self.obs_space[self.dof_obv_key]] if remove_task_state else b[self.dof_obv_key]) for b in batch] 
            tr_actions = [torch.from_numpy(b['action']) for b in batch]

            tr_vobvs_pad = pad_sequence(tr_vobvs, batch_first=True, padding_value=0)
            tr_dof_pad = pad_sequence(tr_dof, batch_first=True, padding_value=0)
            tr_actions_pad = pad_sequence(tr_actions, batch_first=True, padding_value=0)

            padded_batch = {self.vis_obv_key : tr_vobvs_pad, self.dof_obv_key : tr_dof_pad, 'action' : tr_actions_pad}
            return padded_batch

        return pad_collate

    def get_traj_dataloader(self, batch_size, num_workers=1, shuffle=True):
        dataloader = DataLoader(dataset=self.traj_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, collate_fn=self._collate_wrap(remove_task_state=(batch_size != 1)))
        return dataloader
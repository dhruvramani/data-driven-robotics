import re
import os
import sys
import uuid
import torch
import pickle
import tarfile
import numpy as np
import torchvision
from random import randint
from data_config import get_dataset_args, ep_type

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../db/'))
config = get_dataset_args()

from traj_db.models import ArchiveFile

def store_trajectoy(trajectory, episode_type=config.episode_type, task=None):
    ''' 
        Save trajectory to the corresponding database based on env and env_type specified in config.
        + Arguments:
            - trajectory: {deg.vis_obv_key : np.array([n]), deg.dof_obv_key : np.array([n]), 'action' : np.array([n])}
            - episode_type [optional]: tag to store trajectories with (eg. 'teleop' or 'imitation')
    '''
    if 'EPISODE_' not in episode_type:
        episode_type = ep_type(episode_type)
    if task is None:
        task = config.env_type
    assert 'action' in trajectory.keys()

    # NOTE : Current data_path is a placeholder. Edited below with UUID.
    metadata = config.traj_db(task_id=task, env_id=config.env, 
        data_path=config.data_path, episode_type=episode_type, traj_steps=trajectory['action'].shape[0])

    metadata.save()
    metadata.data_path = os.path.join(config.data_path, "traj_{}.pt".format(metadata.episode_id))
    metadata.save()

    with open(metadata.data_path, 'wb') as file:
        pickle.dump(trajectory, file, protocol=pickle.HIGHEST_PROTOCOL)

def get_trajectory(episode_type=None, index=None, episode_id=None):
    '''
        Gets a particular trajectory from the corresponding database based on env and env_type specified in config.
        + Arguments:
            - episode_type [optional]: if you want trajectory specific to one episode_type (eg. 'teleop' or 'imitation')
            - index [optional]: get trajectory at a particular index
            - episode_id [optional]: get trajectory by it's episode_id (primary key)
        
        + NOTE: either index or episode_id should be not None.
        + NOTE: episode_type, env_type become POINTLESS when you pass episode_id.
    '''
    # TODO : If trajectory is in archive-file, get it from there
    if episode_id is None and index is None:
        return [get_trajectory(episode_id=traj_obj.episode_id) for traj_obj in config.traj_db.objects.all()]

    if index is not None:
        if episode_type is None: # TODO : Clean code
            metadata = config.traj_db.objects.filter(task_id=config.env_type)[index] if config.get_by_task_id else config.traj_db.objects.all()[index]
        else:
            metadata = config.traj_db.objects.filter(task_id=config.env_type, episode_type=episode_type)[index] if config.get_by_task_id else config.traj_db.objects.filter(episode_type=episode_type)[index]
    elif episode_id is not None:
        episode_id = str(episode_id)
        metadata = config.traj_db.objects.get(episode_id=uuid.UUID(episode_id))

    with open(metadata.data_path, 'rb') as file:
        trajectory = pickle.load(file)

    return trajectory

def get_random_trajectory(episode_type=None):
    '''
        Gets a random trajectory from the corresponding database based on env and env_type specified in config.
        + Arguments:
            - episode_type [optional]: if you want trajectory specific to one episode_type (eg. 'teleop' or 'imitation')
    '''
    count = config.traj_db.objects.count()
    random_index = randint(1, count)
    if episode_type is None:
        metadata = config.traj_db.objects.filter(task_id=config.env_type)[random_index] if config.get_by_task_id else config.traj_db.objects.all()[random_index]
    else:
        metadata = config.traj_db.objects.filter(task_id=config.env_type, episode_type=episode_type)[random_index] if config.get_by_task_id else config.traj_db.objects.filter(episode_type=episode_type)[random_index]
    
    episode_id = str(metadata.episode_id)
    task_id = metadata.task_id
    trajectory = get_trajectory(episode_id=episode_id)

    return trajectory, episode_id, task_id

def create_video(trajectory):
    '''
        Creates videos and stores video, the initial and the final frame in the paths specified in data_config. 
        + Arguments:
            - trajectory: {deg.vis_obv_key : np.array([n]), deg.dof_obv_key : np.array([n]), 'action' : np.array([n])}
    '''
    frames = trajectory[config.obv_keys['vis_obv_key']].astype(np.uint8)
    assert frames.shape[-1] == 3
    
    inital_obv, goal_obv = Image.fromarray(frames[0]), Image.fromarray(frames[-1])
    inital_obv.save(os.path.join(config.media_dir, 'inital.png'))
    goal_obv.save(os.path.join(config.media_dir, 'goal.png'))

    if type(frames) is not torch.Tensor:
        frames = torch.from_numpy(frames)

    torchvision.io.write_video(config.vid_path, frames, config.fps)
    return config.vid_path

# NOT TESTED
def archive_traj_task(task=config.env_type, episode_type=None, file_name=None):
    ''' 
        Archives trajectories by task (env_type)
        + Arguments:
            - task: config.env_type - group and archive them all.
            - episode_type [optional]: store trajectories w/ same task, episode_type together.
            - file_name: the name of the archive file. [NOTE: NOT THE PATH]
                >  NOTE : default file_name: `env_task.tar.gz`
    '''    
    if episode_type is None:
        objects = config.traj_db.objects.get(task_id=task)
        f_name = "{}_{}.tar.gz".format(config.env, config.env_type)
    else:
        objects = config.traj_db.objects.get(task_id=task, episode_type=episode_type)
        f_name = "{}_{}_{}.tar.gz".format(config.env, config.env_type, episode_type)
    
    if file_name is None:
        file_name = f_name
    file_name = os.path.join(config.archives_path, file_name)

    tar = tarfile.open(file_name, "w:gz")
    for metadata in objects:
        if metadata.is_archived == True:
            continue
            
        metadata.is_archived = True
        metadata.save()
        
        tar.add(metadata.data_path)

        archive = ArchiveFile(trajectory=metadata, env_id=metadata.env_id, archive_file=file_name)        
        archive.save()
    tar.close()

def delete_trajectory(episode_id):
    obj = config.traj_db.objects.get(episode_id=uuid.UUID(episode_id))
    if os.path.exists(obj.data_path):
        os.remove(obj.data_path)
    obj.delete()

def flush_traj_db():
    raise NotImplementedError
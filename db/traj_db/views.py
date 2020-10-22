import os
import sys
import uuid
from django.shortcuts import render

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../dataset_env'))

from data_config import get_dataset_args
import file_storage

config = get_dataset_args()

def vid(request):
    trajectory, episode_id, task_id = file_storage.get_random_trajectory()
    vid_path = file_storage.create_video(trajectory)
    assert os.path.isfile(vid_path)

    return render(request, 'vid.html', {})

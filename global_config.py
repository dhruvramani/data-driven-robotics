import os
import sys
import json
import datetime
import argparse

import utils

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_env/'))

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
DATA_DIR = '/scratch/robotics_data'
TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# NOTE : RLBench  - Run `Xvfb :DISP_NUM -screen 0 1024x768x24 +extension GLX +render -noreset &`
#        `export DISPLAY=:DISP_NUM`, where `DISP_NUM` >= 99 (whichever is free)
# NOTE : RoboSuite - Run `xvfb-run -a -s "-screen 0 1400x900x24" zsh` & 
#        `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` for rendering.

def get_global_parser():
    ''' Global Config - contains global arguments common to all modules.            
        NOTE: All the paths/dirs have env-env_type-exp_name concated to them at the end. 
            - see model/model_config.py for example.
    '''

    parser = argparse.ArgumentParser("Data driven robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env', type=str, default='RLBENCH') #SURREAL
    parser.add_argument('--env_type', type=str, default='close_drawer-vision-v0') #SawyerPickPlace
    parser.add_argument('--env_args', type=env2args, default='RLBENCH') # NOTE : placeholder, changed later.
    parser.add_argument('--exp_name', type=str, default='v0.5')
    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, 'data_files/saved_data/'))
    parser.add_argument('--display_warnings', type=utils.str2bool, default=False)

    return parser

def env2args(env_name):
    # NOTE - WARNING : Changing camera_height, camera_width changes the camer_obs dim. 
    surreal_args = dict(has_renderer=True, has_offscreen_renderer=True, ignore_done=True, use_camera_obs=True,  
        camera_height=256, camera_width=256, camera_name='agentview', use_object_obs=False, reward_shaping=True)

    # NOTE - WARNING : observation_mode='vision' is very heavy
    rlbench_args = dict(use_gym=False, observation_mode='left_shoulder_rgb', vis_obv_key='left_shoulder_rgb', render_mode='rgb_array',
        keyboard_teleop=False, combine_action_space=True)

    env2args_dict = {'SURREAL' : surreal_args, 'RLBENCH' : rlbench_args}
    return env2args_dict[env_name]

def env2deg(env_name):
    from dataset_env.surreal_deg import SurrealDataEnvGroup
    from dataset_env.rlbench_deg import RLBenchDataEnvGroup

    # NOTE: *IMPORTANT* - Use these in model_config & demons_config ONLY.
    # *DEADLOCK* : can lead to deadlock if an object is created in global_config.
    env2deg_dict = {'SURREAL' : SurrealDataEnvGroup,'RLBENCH' : RLBenchDataEnvGroup}

    if env_name is None:
        return None
    return env2deg_dict[env_name.upper()]

if __name__ == '__main__':
    print("=> Testing global_config.py")
    args = get_global_parser().parse_args()
    print(args.env)
    deg = env2deg('SURREAL')
    print(deg.action_space)

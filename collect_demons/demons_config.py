import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import utils
from global_config import *

def get_demons_args():
    parser = get_global_parser()

    # NOTE: 'SURREAL' is a placeholder. The deg is set according to global_config.env -> see below. v
    parser.add_argument('--deg', type=env2deg, default='SURREAL')
    parser.add_argument("--collect_by", type=str, default='teleop', choices=['teleop', 'imitation', 'expert', 'policy', 'exploration', 'random'])
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--collect_freq", type=int, default=1)
    parser.add_argument("--flush_freq", type=int, default=25) # NOTE : RAM Issues, change here : 75
    parser.add_argument("--break_traj_success", type=utils.str2bool, default=True)
    parser.add_argument("--n_runs", type=int, default=10, #10
        help="no. of runs of traj collection, affective when break_traj_success = False")

    # Imitation model
    parser.add_argument('--resume', type=utils.str2bool, default=False)
    parser.add_argument('--train_imitation', type=utils.str2bool, default=False)
    parser.add_argument('--models_save_path', type=str, default=os.path.join(DATA_DIR, 'runs/imitation-models/'))
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(DATA_DIR, 'runs/imitation-tensorboard/'))
    parser.add_argument('--load_models', type=utils.str2bool, default=True)
    parser.add_argument('--use_model_perception', type=utils.str2bool, default=True)
    parser.add_argument('--n_gen_traj', type=int, default=200, help="Number of trajectories to generate by imitation")

    config = parser.parse_args()
    config.env_args = env2args(config.env)
    config.deg = env2deg(config.env)
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.models_save_path = os.path.join(config.models_save_path, '{}_{}/'.format(config.env, config.env_type))
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 

    if config.train_imitation and not config.resume:
        utils.recreate_dir(config.models_save_path, config.display_warnings)
        utils.recreate_dir(config.tensorboard_path, config.display_warnings)
    else:
        utils.check_n_create_dir(config.models_save_path, config.display_warnings)
        utils.check_n_create_dir(config.tensorboard_path, config.display_warnings)

    utils.check_n_create_dir(config.data_path, config.display_warnings)

    return config

if __name__ == '__main__':
    args = get_demons_args()
    print(args.models_save_path)

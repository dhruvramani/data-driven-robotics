# A Boilerplate For Data Driven Robotics
& a Pytorch implementation of [Learning Latent Plans from Play](https://learning-from-play.github.io/).

This repo is supposed to provide organized & scalable experimentation of data-driven robotics learning. You can adapt it to your own model and environment with minor modifications.

## Organization
### Modules
This setup consists of a databse (`db/`) inspired from [[1]](https://arxiv.org/abs/1909.12200) storing meta-data of the trajectories collected and a light web-app which renders a video of the trajectory. The DEG module (`dataset_env/`) provide easy adaption to various environments, provide dataloaders, (`deg_base.py`), easy functionality to interact with the DB and store/retrieve trajectories (`file_storage.py`) - all bundled up. The current implementation includes support for [RLBench](https://github.com/stepjam/RLBench/) and (older)[Robosuite](https://github.com/ARISE-Initiative/robosuite) environments. The collection module (`collect_demons/`) provides data-collection mechanisms such as teleoperation and imitation policies. Every new model can have it's on directory and the current `model/` contains a Pytorch implementation of [LfP](https://learning-from-play.github.io/). The training and testing code is defined in `model/` too.

Additional information about each module is provided in their respective READMEs.
### Configs
Config common to all the modules is defined in `global_config.py`. Each of the other modules have their own config files (`*_config.py`) which add to the global config. The config system is designed to automatically change on minor edits (eg. a change in `env` changes all the paths and other env-related properties).

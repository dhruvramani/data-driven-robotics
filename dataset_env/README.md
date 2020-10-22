# Datasets & Environment Groups

`DataEnvGroup`s (DEGs) provide a common window to interact with environments and their collected datasets. The abstract class (`deg_base.py`) provides most of the common functionalities and has to be overriden to adopt to a new environment. Each environment has it's own DEG file which provides definitions for the abstract methods and properties - see `rlbench_deg.py` and `surreal_deg.py`. Use the DEG to define & get environments, datasets and dataloaders and observation/action spaces with a common environment-agnostic syntax. 

The data-based configs and the link to the DB are defined in `data_config.py`. The environment is specified in `../global_config.py` and it's modification reflects the changes in data-configs. File and DB related functionalities are defined in `file_storage.py`. Visual augmentations to improve performance, inspired from [RAD](https://mishalaskin.github.io/rad/) - are provided in `data_aug.py`. 

## Adding New Environments

All the envs are stored in a directory outside the repo (defined by `ENV_PATH` in each of the DEG files) because of cleanliness. If you want to modify the original  environment, clone it into this directory. All the cloning and renaming code goes in `organize.sh`.

To create a DEG for a new environment, create a new `envname_deg.py` file and inherit the `DataEnvGroup` class. Import the actual environment class and provide definitions for environment-specific abstract methods (like `get_env`, `teleoperate` etc.) and properties (observation space & keys). See `rlbench_deg.py` for example.

After adding the DEG file, create a table for the environment in `db/` (refer to its README). Add the table info in `taj_db_dict` & `env2keys()` in `data_config.py` and add the DEG info in `env2deg()` in `../global_config.py`. That's it! The rest of the functionalities, properties and configs will adopt automatically to the change in `env` in `../global_config.py`.

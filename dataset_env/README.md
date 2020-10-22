# Datasets & Environment Groups (DEGs)

All the envs are stored in the same directory outside the repo (currently my `scratch` dir) because of space constraints. If you want to modify the environment, clone it into this directory. All the cloning (and renaming) code goes in `organize.sh`.

Each env has it's own DEG file, which has a `add_env()` method (call it right after defining) to add the path of the environment and a `DataEnvGroup` subclass providing the definitions of the abstract methods and properties.

The choice of the environment can be specified in `../global_config.py`. Everything changes accordingly without you having to modify each of the properties.

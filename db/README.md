## Django Based Database for Trajectory Meta-data. 

## Setup 
To run the usual commands on `manage.py`, run `python3 manage.py` and enter commands when prompted. I Had to edit `manage.py` out due to errors with argv and the global config. For running `runserver`, type the command twice (as prompted).

Run the commands in order to setup the DB from scratch. 
```
createsuperuser
makemigrations
migrate
```

## Database
The `traj_db` app provides the database to store the meta-data of the trajectories. The `Trajectory` table is an abstract-table. Each environment gets its own table just by inheriting it (see `traj_db/models.py`). Each trajectory is stored as a pickle file and is represented by UUID. The table includes other information such as the task it's performing, how it was generated etc. All the functionalities to interact with the pickle files and the DB is provided in `../dataset_env/file_storage.py`.

## Adding New Environments/Tables
To add a table for a new environment, create a new, empty subclass of `Trajectory` and register it in `traj_db/admin.py`. That's it! Follow `../dataset_env`'s README for other config-modifications. 

## Views
The `admin/` portal provides a good interface to the database, but usage of functions in `../dataset_env/file_storage.py` is recommended. The index page generates a video of a random trajectory of the current environment and displays it. 

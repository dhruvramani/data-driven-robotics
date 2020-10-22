import os
import sys
import shutil
import pathlib

def delete_folder(pth):
    pth = pathlib.Path(pth)
    for sub in pth.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else :
            sub.unlink()
    pth.rmdir()

def delete_shutil(path):
    shutil.rmtree(path)

def recreate_dir(path, display_warning=True):
    if os.path.exists(path):
        delete_shutil(path)
    os.makedirs(path)
    
    if display_warning:
        print("=> Directory {} created.".format(path))

def check_n_create_dir(path, display_warning=True):
    if not os.path.isdir(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    elif display_warning:
        print("=> Directory {} exists (no new directory created).".format(path))

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]
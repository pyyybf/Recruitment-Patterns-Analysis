import os
import shutil


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def list_dir(dir_path):
    file_names = [file_name for file_name in os.listdir(dir_path) if not file_name.startswith(".")]
    return file_names

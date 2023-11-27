import os
import shutil
import json


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def list_dir(dir_path):
    file_names = [file_name for file_name in os.listdir(dir_path) if not file_name.startswith(".")]
    return file_names


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


def save_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

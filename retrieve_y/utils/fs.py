import os
import shutil


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def list_dir(dir_path):
    file_names = [file_name for file_name in os.listdir(dir_path) if not file_name.startswith(".")]
    return file_names


def copy_file(source_file, target_dir, new_file_name=None):
    with open(source_file, "r") as source:
        content = source.read()
    file_name = new_file_name or source_file.split("/")[-1]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(f"{target_dir}/{file_name}", "w") as target:
        target.write(content)

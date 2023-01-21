import GLOBAL_VAR
import PATHS

import yaml
import json
import os
import pickle

import datetime
import string
import secrets

import fnmatch

def load_yaml_file(filename):
    filename = PATHS.EXTRA_PATH+filename
    filename = filename+".yaml" if not (".yaml" in filename) else filename
    with open(filename) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def load_json_file(file_path:str, no_extra=False):
    file_path = file_path if no_extra else PATHS.EXTRA_PATH+file_path
    if not file_path.endswith(".json"):
        file_path += ".json"
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    return data_dict

def load_pickle_file(file_path:str, with_extra=True):
    file_path = PATHS.EXTRA_PATH+file_path if with_extra else file_path
    if not file_path.endswith(".pkl"):
        file_path += ".pkl"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as pickle_file:
            data_dict = pickle.load(pickle_file)
        return data_dict
    else:
        return None


def save_json_dict_to_path(path: str, data: dict):
    path = PATHS.EXTRA_PATH+path
    if GLOBAL_VAR.SAVE:
        if not path.endswith(".json"):
            path += ".json"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

def save_pickle_dict_to_path(path: str, data: dict, is_new=False):
    path = PATHS.EXTRA_PATH+path
    if GLOBAL_VAR.SAVE:
        if path.endswith(".json"): raise Exception("Error: should not end with .json")
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
        if not path.endswith(".pkl"): path += ".pkl"
        already_exists = os.path.exists(path)
        if is_new and already_exists :raise Exception("Error: the path '"+path+"' is not new, but should be.")
        with open(path, 'wb') as file:
            pickle.dump(data, file)


def search_json_in_dir(dir_path):
    dir_path = PATHS.EXTRA_PATH+dir_path
    files = os.listdir(dir_path) if os.path.isdir(dir_path) else []
    json_files = fnmatch.filter(files, "*.json")
    if len(json_files)==0:
        return None
    elif len(json_files) == 1:
        return json_files[0]
    else:
        raise Exception("Error: multiple files found in "+dir_path)

def search_pickle_in_dir(dir_path):
    dir_path = PATHS.EXTRA_PATH+dir_path
    files = os.listdir(dir_path) if os.path.isdir(dir_path) else []
    pickle_files = fnmatch.filter(files, "*.pkl")
    if len(pickle_files)==0:
        return None
    elif len(pickle_files) == 1:
        return pickle_files[0]
    else:
        raise Exception("Error: multiple files found in "+dir_path)


def create_folder(dir_path):
    dir_path = PATHS.EXTRA_PATH+dir_path
    if GLOBAL_VAR.SAVE:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def get_subfolders_names(directory):
    directory = PATHS.EXTRA_PATH+directory
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def get_filenames_in_dir(directory):
    directory = PATHS.EXTRA_PATH+directory
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def create_unique_date_id(length=4):    
    formatted_time = datetime.datetime.now().strftime("%y-%m-%d-%Hh%Mmin%Ssec%f")[:-1]
    unique_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(length))
    return formatted_time+"-"+unique_key

def find_most_recent(dates_list):
    return max(dates_list, key=lambda x: datetime.datetime.strptime(x, "%y-%m-%d-%Hh%Mmin%Ssec%f"))


def retrieve_json_dicts_in_dir(dir_path):
    dir_path = PATHS.EXTRA_PATH+dir_path
    dumped_dict = {}
    filenames_list = get_json_filenames(dir_path)
    for filename in filenames_list:
        with open(dir_path+"/"+filename, 'r') as f:
            new_dict = json.load(f)
            new_dict_dumped = dump_dict(new_dict)
            dumped_dict[filename] = new_dict_dumped
    return dumped_dict


def dump_dict(dict):
    return json.dumps(dict, sort_keys=True)
    

def search_dict_in_dumps(target_dict, dumped_dicts):
    dumped_target_dict = json.dumps(target_dict, sort_keys=True)
    for filename, dumped_dict in dumped_dicts.items():
        if dumped_dict == dumped_target_dict:
            return filename
    return None


def get_json_filenames(directory):
    directory = PATHS.EXTRA_PATH+directory
    filenames = []
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filenames.append(filename)
    return filenames

def get_yaml_filenames(directory):
    directory = PATHS.EXTRA_PATH+directory
    filenames = []
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".yaml"):
                filenames.append(filename)
    return filenames


def list_of_lines2file(list_of_lines, file_path):
    file_path = PATHS.EXTRA_PATH+file_path
    with open(file_path, "w") as f:
        for line in list_of_lines:
            if not line.endswith("\n"): line += "\n"
            f.write(line)
    

def get_tree_structure(path, first_call=True):
    if first_call: path = PATHS.EXTRA_PATH+path
    tree = {}
    for item in os.listdir(path):
        full_item_path = os.path.join(path, item)
        if os.path.isdir(full_item_path):
            # If the item is a directory, recurse into it and add the resulting
            # tree structure to the current tree.
            tree[item] = get_tree_structure(full_item_path, first_call=False)
        else:
            # If the item is a file, add it to the tree.
            tree[item] = None
    return tree
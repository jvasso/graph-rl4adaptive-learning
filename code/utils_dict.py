import utils_files

import json


def get_value_at_path(path, tree:dict):
    assert type(tree)==dict
    path = maybe_str_path2list_path(path)
    
    # Initialize the current node with the root of the tree
    current_node = tree

    # Iterate over the elements in the path
    for element in path:
        # If the current node is not a dictionary or if the element is not
        # a valid key in the current node, return None
        if not isinstance(current_node, dict) or element not in current_node:
            return None

        # Update the current node to be the value associated with the element
        current_node = current_node[element]

    # Return the current node (which should be the value at the end of the path)
    return current_node


def set_value_at_path(path, value, tree:dict, new_path=False):
    assert type(tree)==dict
    path = maybe_str_path2list_path(path)
    current_dict = tree
    for i in range(len(path)):
        full_key_str = str("/".join( [path[j] for j in range(i)] ))
        key = path[i]
        if key in current_dict:
            if i == len(path)-1:
                if new_path:
                    raise Exception("Error: the path '"+path+"' already exists in this dictionary.")
                current_dict[key] = value
            elif type(current_dict[key]) == dict:
                current_dict = current_dict[key]
            else:
                print("WARNING: Parameter "+full_key_str+" is neither a leaf nor a dict.")
                break
        elif new_path:
            if i == len(path)-1:
                current_dict[key] = value
            else:
                current_dict[key] = {}
        else:
            raise Exception("Error: Parameter "+full_key_str+" was not found in hyperparameters.")
    return tree


def maybe_str_path2list_path(path):
    if type(path) == str:
        return str_path2list_path(path)
    elif type(path) == list:
        return path
    else:
        raise Exception("path object must be a list or str")


def str_path2list_path(str_path:str):
    # cannot start with "/"
    if str_path.startswith("/"):
        str_path = str_path[1:]
    
    path = str_path.split("/")
    return path


def merge_disjointed_dicts(*dicts):
    # assert dictionaries do not have common keys
    common_keys = set(dicts[0].keys())
    for dict_ in dicts[1:]:
        common_keys &= set(dict_.keys())
    if len(common_keys)>0:
        raise Exception("This function can only merge disjointed dictionaries.")
    # merge
    merged_dict = {}
    for dict_ in dicts:
        merged_dict.update(dict_)
    return merged_dict


def LD2DL(LD):
    DL = {k: [dic[k] for dic in LD] for k in LD[0]}
    return DL

def connect_dict_to_file(input_dict_path:str, output_os_path:str, data_dict:dict):
    target_filename = get_value_at_path(input_dict_path, data_dict)
    target_file_path = output_os_path + "/" + target_filename
    target_dict = utils_files.load_yaml_file(target_file_path)
    assert target_dict is not None
    set_value_at_path(input_dict_path, target_dict, data_dict)


def connect_dict_internally(input_dict_path:str, output_dict_path:str, data_dict:dict):
    target_filename = get_value_at_path(input_dict_path, data_dict)
    target_file_path = output_dict_path + "/" + target_filename
    target_dict = get_value_at_path(target_file_path, data_dict)
    set_value_at_path(output_dict_path, target_dict, data_dict)


def print_dict_pretty(data_dict:dict):
    pretty = json.dumps(data_dict, indent=5)
    print(pretty)


def remove_duplicates(tree_dict):
    for key in tree_dict:
        if isinstance(tree_dict[key], list):
            tree_dict[key] = list(set(tree_dict[key]))
        elif isinstance(tree_dict[key], dict):
            remove_duplicates(tree_dict[key])


def customized_dumps(tree_dict, indent=4):
    def tree_to_json(d, indent):
        if isinstance(d, dict):
            space = ' '*indent
            return '{\n'+space + ',\n'.join('"{}": {}'.format(k, tree_to_json(v, indent)) for k, v in d.items()) + '\n}'
        elif isinstance(d, list):
            return '[' + ', '.join(tree_to_json(v, indent) for v in d) + ']'
        else:
            return json.dumps(d, indent=indent)
    return tree_to_json(tree_dict, indent)

def tree_dict_list2str(tree):
    for key, value in tree.items():
        if isinstance(value, list):
            tree[key] = str(value)
        elif isinstance(value, dict):
            tree_dict_list2str(value)
    return tree

def remove_spurious_patterns(text_dict):
    text_dict = replace_pattern(text_dict, '"[', '[')
    text_dict = replace_pattern(text_dict, ']"', ']')
    return text_dict

def replace_pattern(text, pattern1, pattern2):
    return text.replace(pattern1, pattern2)


def flatten_tree(tree):
    for key in tree:
        if isinstance(tree[key], dict):
            flatten_tree(tree[key])
        elif isinstance(tree[key], list) and len(tree[key]) == 1:
            tree[key] = tree[key][0]
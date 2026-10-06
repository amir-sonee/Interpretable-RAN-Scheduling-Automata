import os
import shutil
import numpy as np
import json


def get_param(param_dict, param_name, default_value=None):
    if param_dict is not None and param_name in param_dict:
        return param_dict[param_name]
    return default_value


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rm_dir(dir_name):
    if path_exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)


def rm_dirs(dir_list):
    for dir_name in dir_list:
        rm_dir(dir_name)


def rm_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def rm_files(file_list):
    for filename in file_list:
        rm_file(filename)


def is_file_empty(path):
    return os.path.getsize(path) == 0


def path_exists(path):
    return os.path.exists(path)


def sort_by_ord(input_list):
    input_list.sort(key=lambda s: ord(s.lower()))


def randargmax(input_vector):
    return np.random.choice(np.flatnonzero(input_vector == np.max(input_vector)))


def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


def write_json_obj(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def write_json_obj_pretty(obj, filepath):
    with open(filepath, 'w') as f:
        dump_outer_pretty_inner_inline(obj, f, indent=4)


def dump_outer_pretty_inner_inline(data, fp, indent=4):
    def write(obj, level, in_outer_list=False):
        if isinstance(obj, dict):
            fp.write('{\n')
            items = list(obj.items())
            for i, (k, v) in enumerate(items):
                fp.write(' ' * ((level + 1) * indent))
                fp.write(json.dumps(k))
                fp.write(': ')
                write(v, level + 1, in_outer_list=True)
                if i != len(items) - 1:
                    fp.write(',')
                fp.write('\n')
            fp.write(' ' * (level * indent) + '}')
        elif isinstance(obj, list):
            if in_outer_list:
                # Pretty-print outer list, inline its children
                fp.write('[\n')
                for i, v in enumerate(obj):
                    fp.write(' ' * ((level + 1) * indent))
                    fp.write(json.dumps(v, separators=(',', ': ')))  # inner list inline
                    if i != len(obj) - 1:
                        fp.write(',')
                    fp.write('\n')
                fp.write(' ' * (level * indent) + ']')
            else:
                # Not in outer list → inline
                fp.write(json.dumps(obj, separators=(',', ': ')))
        else:
            fp.write(json.dumps(obj))

    write(data, 0)

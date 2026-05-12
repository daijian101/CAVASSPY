import os

from scipy.io import loadmat
from scipy.io import savemat


def read_mat(input_file, key='scene'):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found.')
    data = loadmat(input_file)[key]
    return data


def save_mat(output_file, data, key='scene'):
    ensure_output_file_dir_existence(output_file)
    savemat(output_file, {key: data})


def ensure_output_dir_existence(output_dir):
    mk_output_dir = not os.path.exists(output_dir)
    if mk_output_dir:
        os.makedirs(output_dir)
    return mk_output_dir, output_dir


def ensure_output_file_dir_existence(output_file):
    output_dir = os.path.split(output_file)[0]
    return ensure_output_dir_existence(output_dir)

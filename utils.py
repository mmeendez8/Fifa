import os
import shutil
import numpy as np


def create_results_folder(results_folder='Results'):
    for folder in ['Test', 'Train']:
        if not os.path.exists(os.path.join(results_folder, folder)):
            os.makedirs(os.path.join(results_folder, folder))
    return results_folder


def create_folder(folder):
    if os.path.exists(folder):
        delete_old_logs(folder)
        os.makedirs(folder)
    return folder


def get_model_description(flags):
    description = ''
    flag_names = ['latent_dim', 'learning_rate']
    for key in flags.__flags.keys():
        if key in flag_names:
            description += '{}_{}_'.format(key, getattr(flags, key))

    return description[:-1]


def delete_old_logs(logdir):
    try:
        shutil.rmtree(logdir)
    except:
        return


def get_files(base_dir):
    files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    return files


def get_interp(v1, v2, n):
    ''' Sample points for interpolation of vectors'''
    if not v1.shape == v2.shape:
        raise Exception('Different vector size')

    return np.array([np.linspace(v1[i], v2[i], n+2) for i in range(v1.shape[0])]).T



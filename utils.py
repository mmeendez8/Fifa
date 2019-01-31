import os
import shutil


def create_results_folder(results_folder='Results'):
    for folder in ['Test', 'Train']:
        if not os.path.exists(os.path.join(results_folder, folder)):
            os.makedirs(os.path.join(results_folder, folder))
    return results_folder


def get_model_description(flags):
    description = ''
    flag_names = ['latent_dim', 'learning_rate']
    for f in flag_names:
        description += '{}_{}_'.format(f, flags[f])

    return description[:-1]


def delete_old_logs(logdir):
    try:
        if not len(os.listdir(logdir)) == 0:
            shutil.rmtree(logdir)
    except:
        return


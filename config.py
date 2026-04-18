from os.path import join, dirname, exists
from os import getcwd, listdir, makedirs, cpu_count
from itertools import product
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

DIR_PATH = dirname(join(getcwd(), __file__))
INPUT_PATH = join(DIR_PATH, 'dataset')
LOG_NAME = sorted([item.split('_processed.g')[0] for item in listdir(INPUT_PATH)
                   if item.endswith('_processed.g')])
MAX_WORKERS = cpu_count()

TRAIN_SPLIT = 0.67
PATIENCE = 20
BATCH_SIZE = 64
DROPOUT = 0.1

# complete grid search
# """
EPOCHS = 200
#LEARNING_RATE = [1e-4]
LEARNING_RATE = [1e-3, 1e-4]
#HIDDEN_LAYERS = [0]
HIDDEN_LAYERS = [0, 1, 2, 5]
#HEADS = [2]
HEADS = [1, 2]
LAYERS_SIZE = [64]
#K = [200]
K = [50, 100, 200]
VARIANT_TO_TEST = ['var_2gnn']
# """


def get_result_path(log_name, variant, k):
    result_path = join(DIR_PATH, 'results', variant, f'{log_name}_{k}_k')
    makedirs(result_path, exist_ok=True)
    return result_path


def get_grid_combinations(log_name, variant, k):
    grid_search = []
    result_path = get_result_path(log_name, variant, k)
    combinations_done = [comb for comb in listdir(result_path) if exists(join(result_path, comb, 'done'))]
    for layers_size, hidden_layers, learning_rate, heads in product(LAYERS_SIZE, HIDDEN_LAYERS, LEARNING_RATE, HEADS):
        data_comb = f'{learning_rate}_lr_{hidden_layers}_l_{layers_size}_s_{heads}_h'
        if data_comb not in combinations_done:
            parameters = {
                'learning_rate': learning_rate,
                'hidden_layers': hidden_layers,
                'layers_size': layers_size,
                'heads': heads,
            }

            grid_search.append((join(result_path, data_comb), data_comb, parameters))
    return grid_search


def resume_grid_combination(combination_path):
    to_resume = not exists(join(combination_path, 'done')) and exists(join(combination_path, 'checkpoint.tar'))
    return to_resume

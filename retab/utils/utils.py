import random
import numpy as np
import torch
from addict import Dict
import logging
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml
import json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_dict(d: dict):
    for k, v in d.items():
        print(f"{k}: {v}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def make_logger(filename, logger_name=None):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_config(config_path):
    with open(config_path, 'r') as file:
        return Dict(yaml.safe_load(file))


def get_params(cfg, key=None, trial=None, HPO=False):
    if trial==None and HPO==True:
        raise ValueError("If HPO=True, Optuna object trial should be given.")
    
    params = Dict({})
    if HPO:
        for name, hp_config in cfg.hyperparameters[key].items():
            if hp_config['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, hp_config.grid)
            elif hp_config['type'] == 'int':
                params[name] = trial.suggest_int(name, hp_config.low, hp_config.high)
            elif hp_config['type'] == 'float':
                params[name] = trial.suggest_float(name, hp_config.low, hp_config.high)
    
    for name, setting in cfg[key].items():
        params[name] = setting

    return params


def save_best_cfg_as_yaml(cfg: Dict, best_params, save_path):
    '''
    - replace cfg.hyperparameters with best_params
    - save as yaml file.
    '''
    best_cfg = cfg.copy()
    # delete arguments for hpo setting
    del best_cfg.study 

    model_keys = set(cfg.hyperparameters.model_parameters.keys())
    data_keys = set(cfg.hyperparameters.data_parameters.keys())
    # remove hyperparameters section
    del best_cfg.hyperparameters

    # Split best_params into model_parameters and data_parameters
    for k, v in best_params.items():
        if k in model_keys:
            best_cfg.model_parameters[k] = v
        elif k in data_keys:
            best_cfg.data_parameters[k] = v
        else:
            pass

    # addict to dict
    best_cfg = best_cfg.to_dict()

    # save as yaml
    with open(save_path, 'w') as file:
        yaml.dump(best_cfg, file, default_flow_style=False)
    
    print(f"saving complete. location: {save_path}")


def find_and_replace_key(dictionary, target_key, new_value):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if key == target_key and value != new_value:
                print(f"replace {key}: {value} -> {key}: {new_value}")
                dictionary[key] = new_value
                print("complete.")
            elif isinstance(value, (dict, list)):
                find_and_replace_key(value, target_key, new_value)


def args_to_dict(args):
    dictionary = Dict({})
    for k, v in vars(args).items(): 
        dictionary[k] = v
    return dictionary


def generate_abbreviations(keys: dict):
    abbrevs = {}
    used = set()

    for key in keys:
        # split key by '_' and make abbreviation
        parts = key.split('_')
        abbr = ''.join([p[0] for p in parts])
        
        # when key exists, add number to make it unique
        candidate = abbr
        i = 1
        while candidate in used:
            candidate = f"{abbr}{i}"
            i += 1
        
        abbrevs[key] = candidate
        used.add(candidate)

    return abbrevs


def get_exp_id(data_params, model_params, abbrevs_path=None):
    # Merge both dicts, sort by key, and concatenate key-value pairs
    merged = {**data_params, **model_params}
    abbrevs = generate_abbreviations(merged.keys())
    abbrevs = {k: abbrevs[k] for k in sorted(merged.keys())}

    if abbrevs_path:
        with open(abbrevs_path, 'w') as f:
            json.dump(abbrevs, f, indent=4, ensure_ascii=False)
    exp_id = '-'.join(f"{abbrevs[k]}={merged[k]}" for k in sorted(merged.keys()))
    return exp_id
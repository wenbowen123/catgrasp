'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os


def key_to_attr(args,cfg):
    for k in cfg.keys():
        if isinstance(cfg[k],dict):
            key_to_attr(args,cfg[k])
        else:
            setattr(args, k, cfg[k])
    return args

def get_parser(config_dir=f'{os.path.dirname(os.path.realpath(__file__))}/../config/config_pointgroup.yaml'):
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    args_cfg = parser.parse_args()
    args_cfg.config = config_dir
    print('config_dir',config_dir)

    assert os.path.exists(args_cfg.config), f'args_cfg.config: {args_cfg.config}'
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    args_cfg = key_to_attr(args_cfg,config)
    return args_cfg


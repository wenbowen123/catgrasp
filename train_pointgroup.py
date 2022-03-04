import open3d as o3d
import sys
import os
from multiprocessing import cpu_count
import argparse
import torch
from torch import optim
from torch.utils import data
import numpy as np
import yaml
import glob
import random
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/PointGroup')
from trainer_pointgroup import *
from Utils import *


if __name__ == '__main__':
	code_dir = os.path.dirname(os.path.realpath(__file__))
	cfg_dir = '{}/config/config_pointgroup.yaml'.format(code_dir)
	with open(cfg_dir, 'r') as ff:
		cfg = yaml.safe_load(ff)

	random_seed = cfg['random_seed']
	np.random.seed(random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed_all(random_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	code_dir = os.path.dirname(os.path.realpath(__file__))
	save_dir = f'{code_dir}/logs/{cfg["class_name"]}_seg'
	os.system('rm -rf {} && mkdir -p {}'.format(save_dir,save_dir))
	cfg['save_dir'] = save_dir

	with open(f'{save_dir}/config_pointgroup.yaml','w') as ff:
		yaml.safe_dump(cfg,ff)

	trainer = TrainerPointGroup(cfg)
	trainer.train()


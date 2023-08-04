import open3d as o3d
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint
import pdb

cfg = CN()

# experiment name
cfg.exp_name = 'hello'

# network
cfg.point_feature = 9
cfg.distributed = False

# data
cfg.human = 313
cfg.training_view = [0, 7, 15]
# cfg.training_view = [15]
cfg.intv = 1
cfg.begin_i = 0  # the first smpl
cfg.ni = 1  # number of smpls
cfg.i = 1  # the i-th smpl
cfg.i_intv = 1
cfg.nv = 6890  # number of vertices
cfg.smpl = 'smpl_4views_5e-4'
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.mask_bkgd = True
cfg.sample_smpl = False
cfg.sample_grid = False
cfg.sample_fg_ratio = 0.7
cfg.H = 1024
cfg.W = 1024
cfg.add_pointcloud = False

cfg.big_box = False

cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

# mesh
cfg.mesh_th = 50  # threshold of alpha

# task
cfg.task = 'nerf4d'

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True
cfg.specified_resume = ''

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN(
    {'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0

cfg.train.scheduler = CN(
    {'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0, 0, 0, 0, 0]

cfg.train.use_ssim = False
cfg.train.use_d = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN(
    {'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 20
cfg.record_interval = 20

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False
cfg.fix_random = False
cfg.vis = 'mesh'

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(
        [str(gpu) for gpu in cfg.gpus])
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task,
                                         cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.exp_name = cfg.exp_name.replace(',', '_')
    
    parse_cfg(cfg, args)
    # pprint.pprint(cfg)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none',
                    choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if len(args.type) > 0:
    cfg.task = "run"

 
cfg = make_cfg(args)

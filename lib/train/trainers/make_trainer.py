from .trainer import Trainer
import imp

def _wrapper_factory(cfg, network):
    module = cfg.trainer_module
    path = cfg.trainer_path # lib.train.trainers.if_nerf_clight lib/train/trainers/if_nerf_clight.py
    network_wrapper = imp.load_source(module, path).NetworkWrapper(cfg, network)
    return network_wrapper

def make_trainer(cfg, network):

    network = _wrapper_factory(cfg, network)
 
    return Trainer(network)

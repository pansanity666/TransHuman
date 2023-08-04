from lib.config import cfg, args
import numpy as np 

import sys 
sys.path.append('./pytorch3d')

def run_evaluate():
    # metric evaluation 
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    import re 
    
    if cfg.test.full_eval: 
        cfg.test.exp_folder_name += 'full_eval'

    cfg.flag_train = False
    cfg.perturb = 0
    
    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    if k == 'tar_img_path':
                        continue
                    if k == 'input_img_paths':
                        continue    
                    if k == 'human_name':
                        continue
                    if isinstance(batch[k], tuple) or isinstance(batch[k],
                                                                 list):
                        batch[k] = [b.cuda() for b in batch[k]]
                    else:
                        batch[k] = batch[k].cuda()
            
            # Fast rendering 
            output = renderer.render_fast(batch, is_train=False)

            evaluator.evaluate(output, batch)
    evaluator.summarize()

def run_visualize():
    # video visualization 
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    
    for batch in tqdm.tqdm(data_loader):
        
        # batch to cuda 
        for k in batch:
            if k == 'img_path':
                continue
            if k == 'input_img_paths':
                continue    
            if k == 'tar_img_path':
                continue
            if k == 'human_name':
                continue
            if k != 'meta':
                if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                    batch[k] = [b.cuda() for b in batch[k]]
                else:
                    batch[k] = batch[k].cuda()
        
        ### Visualize input images. 
        # import torchvision
        # import os 
        # # import torch.distributed as dist 
        # cached_pth = cfg.trained_model_dir.replace('trained_model', 'cached_imgs')
        # os.makedirs(cached_pth, exist_ok=True)
        # # torchvision.utils.save_image(batch['input_imgs'][0][0], cached_pth + '/E_{}_it_{}_rank_{}.jpg'.format(epoch, iteration, dist.get_rank()))
        # torchvision.utils.save_image(batch['input_imgs'][0][0], cached_pth + '/batch_vos.jpg')

        with torch.no_grad():
            output = renderer.render_fast(batch)
            visualizer.visualize(output, batch)

def run_reconstruction():
    # mesh reconstuction 
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    

    for batch in tqdm.tqdm(data_loader):
        # batch to cuda 
        for k in batch:
            if k == 'img_path':
                continue
            if k == 'input_img_paths':
                continue    
            if k == 'tar_img_path':
                continue
            if k == 'human_name':
                continue
            if k != 'meta':
                if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                    batch[k] = [b.cuda() for b in batch[k]]
                else:
                    batch[k] = batch[k].cuda()
     
        with torch.no_grad():
            # mesh reconstruction does not support fast rendering ATM.
            output = renderer.render(batch)
            visualizer.visualize(output, batch)
            
            
def run_light_stage():
    from lib.utils.light_stage import ply_to_occupancy
    ply_to_occupancy.ply_to_occupancy()


if __name__ == '__main__':
    globals()['run_' + args.type]()

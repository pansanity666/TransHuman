import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg
from torch import nn
import os 

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

class Trainer(object):
    def __init__(self, network):
        
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        
        if cfg.distributed:
            if has_batchnorms(network):
                print('Switching to SyncBN...')
                network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
            print(device)
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True
            )

        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):

        for k in batch:
            if k == 'meta':
                continue
            if k == 'tar_img_path':
                continue
            if k == 'input_img_paths':
                continue    
            if k == 'human_name':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)

        return batch 

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()   

        for iteration, batch in enumerate(data_loader):
            
            # import torchvision
            # import torch.distributed as dist 
            # cached_pth = cfg.trained_model_dir.replace('trained_model', 'cached_imgs')
            # os.makedirs(cached_pth, exist_ok=True)
            # torchvision.utils.save_image(batch['input_imgs'][0][0], cached_pth + '/E_{}_it_{}_rank_{}.jpg'.format(epoch, iteration, dist.get_rank()))

            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)

            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (
                            max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(
                    ['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string,
                                                    str(recorder), lr,
                                                    memory)
                print(training_state)
            
            if cfg.use_record:
                if iteration % cfg.record_interval == 0 or iteration == (
                        max_iter - 1):
                    # record loss_stats and image_dict
                    recorder.update_image_stats(image_stats)
                    recorder.record('train')
                    
            cfg.global_iter += 1 
    

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

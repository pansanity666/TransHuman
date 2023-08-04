from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR, PolynomialLR, WarmupLR
from torch.optim.lr_scheduler import CosineAnnealingLR

def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler

    # if cfg_scheduler.type == 'multi_step':
    #     scheduler = MultiStepLR(optimizer,
    #                             milestones=cfg_scheduler.milestones - cfg_scheduler.warmup_epochs,
    #                             gamma=cfg_scheduler.gamma)
    # elif cfg_scheduler.type == 'exponential':
    #     scheduler = ExponentialLR(optimizer,
    #                               decay_epochs=cfg_scheduler.decay_epochs - cfg_scheduler.warmup_epochs,
    #                               gamma=cfg_scheduler.gamma)
    # elif cfg_scheduler.type == 'poly':
    #     scheduler = PolynomialLR(optimizer,
    #                              total_iters=cfg_scheduler.decay_epochs - cfg_scheduler.warmup_epochs, 
    #                              power=1.0)
    
    if cfg_scheduler.type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                 T_max=cfg_scheduler.decay_epochs - cfg_scheduler.warmup_epochs, 
                                 eta_min=cfg_scheduler.end_lr)

    if cfg_scheduler.warmup_epochs > 0:
        # Warmup Wrapper
        scheduler = WarmupLR(scheduler, init_lr=0.0, num_warmup=cfg_scheduler.warmup_epochs, warmup_strategy='linear')

    ### Visualize Learning Rate. 
    # lrs = []
    # for i in range(cfg_scheduler.decay_epochs):
    #     lrs.append(scheduler.get_lr()[0])
    #     scheduler.step()
    # import matplotlib.pyplot as plt 
    # plt.plot(list(range(cfg_scheduler.decay_epochs)), lrs)
    # plt.savefig('./lr.jpg', )
    # assert False

    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    scheduler.gamma = cfg_scheduler.gamma

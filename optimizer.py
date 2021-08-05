import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
from torch_optimizer import Lamb
import math

"""
https://github.com/jeonsworld/ViT-pytorch:

    - warmupLinearSchedule

"""
class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def optim(model_name, model, lr):
    """
    The script tries to replicates the optimizer used in the paper, sometimes it resulted
    in Nan loss and I've modified the learning rate scheduler to make it work.
    """
    if model_name == 'resnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return optimizer, scheduler
    if model_name == 'alexnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")
        return optimizer, scheduler
    if model_name == 'vggnet':
        """THIS SETTING DOESN'T WORK, RESULTS IN NAN"""
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")
        return optimizer, scheduler
    if model_name == 'vit':
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=0.1)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=10000)
        return optimizer, scheduler

    if model_name == 'mlpmixer':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=10000)
        return optimizer, scheduler

    if model_name == 'resmlp':
        optimizer = Lamb(model.parameters(), lr=lr, weight_decay=0.2)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=10000)
        return optimizer, scheduler
    
    if model_name == 'squeezenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=10000)
        return optimizer, scheduler
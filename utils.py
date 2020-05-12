import torch
import torch.nn as nn
import random
import numpy as np


def gen_z(batch_size, d):
    return torch.randn(batch_size, d)


def gen_yk(batch_size, num_classes):
    t = torch.zeros(batch_size, num_classes)
    for i, row in enumerate(t):
        num = random.randint(0, num_classes - 1)
        # num = i % num_classes
        row[num] = 1
    return t


def fake_v(batch_size, num_classes):
    v = torch.zeros(batch_size) + num_classes
    return v


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=80):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

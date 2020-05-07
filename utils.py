import torch
import torch.nn as nn
import random
import numpy as np


def gen_z(batch_size, d):
    """
    Produces a batch of random normal vectors
    :param batch_size: batch_size
    :param d: size of the normal vector
    :return: batch of random normal vectors
    """
    return torch.rand(batch_size, d)


def gen_yk(batch_size, num_classes):
    """

    :param batch_size:
    :param num_classes:
    :return:
    """
    t = torch.zeros(batch_size, num_classes)
    l = torch.zeros(batch_size)
    for i, row in enumerate(t):
        num = random.randint(0, num_classes - 1)
        row[num] = 1
        l[i] = num
    return l, t


def fake_v(batch_size, num_classes):
    """

    :param batch_size:
    :param num_classes:
    :return:
    """
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

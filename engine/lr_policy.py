#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/1 上午1:50
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : lr_policy.py.py

from abc import ABCMeta, abstractmethod
import math

#Lr_shce
class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)

class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / (self.total_iters*5)) ** self.lr_power)
        #lr reduce slower is better

class CosinLR():
    def __init__(self, start_lr, Te, total_iters, niters_per_epoch):
        self.start_lr = start_lr
        self.max_lr = start_lr
        self.min_lr = 0
        self.total_iters = total_iters + 0.0
        self.niters_per_epoch = niters_per_epoch
        self.Te = Te

    def get_lr(self, cur_iter,cur_epoch):
        cur_Te_iter = cur_iter-(cur_epoch//self.Te)*self.Te * self.niters_per_epoch
        #print("cur_Te_iter",cur_Te_iter)
        new_lr = self.min_lr + (self.max_lr - self.min_lr)*(1+math.cos(math.pi * cur_Te_iter/(self.Te*self.niters_per_epoch)))/2
        return new_lr

class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr



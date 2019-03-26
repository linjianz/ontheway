#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2019-02-25 09:36:58
Program: 
Description: 
"""

import sys
import torch
import numpy as np
import torch.nn as nn
from operator import add, truediv
from torch.autograd import Variable


def to_var(x, requires_grad=False):
	"""

	:param x: numpy array or torch tensor
	:param requires_grad:
	:return:
	"""
	
	if not torch.is_tensor(x):
		x = torch.from_numpy(x)
	if torch.cuda.is_available():
		return Variable(x, requires_grad=requires_grad).cuda()
	else:
		return Variable(x, requires_grad=requires_grad)


def get_parameter_number(net):
	params = list(net.parameters())
	k = 0
	for i in params:
		l = 1
		for j in i.size():
			l *= j
		k = k + l
	return k


def str2bool(v):
	"""bool type for arg parse"""
	
	if v.lower() in ['true', 't', 'yes', 'y']:
		return True
	elif v.lower() in ['false', 'f', 'no', 'n']:
		return False
	else:
		raise Exception('Unsupported value encountered.')


class MeterSingle(object):
	"""Store single value in training process

	Examples::

		>>> loss_meter = MeterSingle()
		>>> loss_meter.update(loss_of_this_step)
	"""
	val: float
	avg: float
	sum: float
	cnt: int
	
	def __init__(self):
		self.reset()
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.cnt = 0


class MeterMulti(object):
	"""Store value list in trn/dev/tst process

	Examples::

		>>> loss_meter = MeterMulti(name_list=['loss', 'acc'], phase_list=['trn', 'dev', 'tst'])
		>>> loss_meter.update(val_list=[loss_tmp, acc_tmp], phase='trn', n=[1, 1])
		# get dev's acc avg
		>>> loss_meter.avg['dev'][1]

	"""
	val: dict
	sum: dict
	avg: dict
	cnt: dict
	
	def __init__(self, name_list, phase_list):
		self.name_list = name_list
		self.phase_list = phase_list
		self.name_num = len(self.name_list)
		self.phase_num = len(self.phase_list)
		self.reset()
	
	def update(self, val_list, phase, n=list()):
		if not n:
			n = self.name_num * [1]
		else:
			assert len(n) == self.name_num, 'count list length not match.'
		
		assert len(val_list) == self.name_num, 'list length not match.'
		assert phase in self.phase_list, 'phase must choose in {}'.format(self.phase_list)
		
		self.val[phase] = val_list
		self.sum[phase] = list(map(add, self.sum[phase], self.val[phase]))
		self.cnt[phase] = list(map(add, self.cnt[phase], n))
		self.avg[phase] = list(map(truediv, self.sum[phase], self.cnt[phase]))
	
	def reset(self):
		self.val = {phase: [0] * self.name_num for phase in self.phase_list}
		self.sum = {phase: [0] * self.name_num for phase in self.phase_list}
		self.avg = {phase: [0] * self.name_num for phase in self.phase_list}
		self.cnt = {phase: [0] * self.name_num for phase in self.phase_list}


class AdjustLR(object):
	def __init__(self, lr_base, interval, endurance):
		self.lr_base = lr_base
		self.interval = interval
		self.endurance = endurance
		
		self.lr_now = lr_base
		self.last_change_lr = 0
	
	def adjust_learning_rate(self, optimizer, epoch, epoch_best):
		"""如果interval个epoch内效果没超过最好的epoch，则降低lr，且endurance内不再改变

		epoch: 1, 2, 3, 4, 5, 6, 7, 8
		best:  1, 2, 2, 2, 2, 2, 2, 2
		last:  0,             6

		:param optimizer:
		:param epoch: 当前epoch
		:param epoch_best: 最好的epoch
		:return:
		"""
		
		if epoch - epoch_best >= self.interval and epoch - self.last_change_lr >= self.endurance:
			for param_group in optimizer.param_groups:
				self.lr_now /= 3.
				param_group['lr'] = self.lr_now
			self.last_change_lr = epoch


class ScheduledOptim(object):
	"""A simple wrapper class for learning rate scheduling"""
	
	def __init__(self, optimizer, d_model, n_warmup_steps):
		self._optimizer = optimizer
		self.n_warmup_steps = n_warmup_steps
		self.n_current_steps = 0
		self.init_lr = np.power(d_model, -0.5)
	
	def step_and_update_lr(self):
		"""Step with the inner optimizer"""
		
		self._update_learning_rate()
		self._optimizer.step()
	
	def zero_grad(self):
		"""Zero out the gradients by the inner optimizer"""
		
		self._optimizer.zero_grad()
	
	def _get_lr_scale(self):
		return np.min([
			np.power(self.n_current_steps, -0.5),
			np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
	
	def _update_learning_rate(self):
		"""Learning rate scheduling per step"""
		
		self.n_current_steps += 1
		lr = self.init_lr * self._get_lr_scale()
		
		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr


class Logger(object):
	def __init__(self, file="default.log"):
		self.terminal = sys.stdout
		self.log = open(file, "w")
	
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
	
	def flush(self):
		pass


def my_loss_compare():
	import torch.nn as nn
	
	nn.BCELoss()
	# 多标签
	# input:	BxC		必须先经过sigmoid归一至[0, 1]，每行表示该样本属于每一类的概率
	# target: 	BxC		所有数字都是0或者1，表示该样本的标签
	
	nn.BCEWithLogitsLoss()
	# 多标签 sigmoid + BCELoss
	# input:	BxC
	# target: 	BxC
	
	nn.NLLLoss()
	# 单标签
	# input:	BxC		必须先经过logsoftmax，（如果用softmax或者sigmoid都会使得误差为负 x）
	# target: 	B		类别编号，0 ~ C-1 的整数
	
	nn.CrossEntropyLoss()  # 等价于tf的sparse_softmax_cross_entropy_with_logits


# 单标签 logsoftmax + NLLLoss
# input:	BxC 	just logits
# target: 	B


##################################################
# TODO: Neural Network Helper
##################################################


class AttrProxy(object):
	"""
	Translates index lookups into attribute lookups.
	To implement some trick which able to use list of nn.Module in a nn.Module
	see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
	"""
	
	def __init__(self, module, prefix):
		self.module = module
		self.prefix = prefix
	
	def __getitem__(self, i):
		return getattr(self.module, self.prefix + str(i))


class ListModule(object):
	"""Layers defined in list can't be recognized by Pytorch, this implementation can make it.

	Examples::

		# in __init__ function
		>>> fc_list = ListModule(self, 'fc_')
		>>> for i in range(3):
		>>>    fc_list.append(fc(100, 128))

		# in forward function
		>>> x = np.random.rand(32, 100)
		>>> x_fcs = [fc_layer(x) for fc_layer in fc_list]
	"""
	
	def __init__(self, module, prefix, *args):
		self.module = module
		self.prefix = prefix
		self.num_module = 0
		for new_module in args:
			self.append(new_module)
	
	def append(self, new_module):
		if not isinstance(new_module, nn.Module):
			raise ValueError('Not a Module')
		else:
			self.module.add_module(self.prefix + str(self.num_module), new_module)
			self.num_module += 1
	
	def __len__(self):
		return self.num_module
	
	def __getitem__(self, i):
		if i < 0 or i >= self.num_module:
			raise IndexError('Out of bound')
		return getattr(self.module, self.prefix + str(i))


def conv(bn, c_in, c_out, ks=(3,), sd=1):
	""" Merge Conv, BN, Relu together

	Args:
		bn		Apply batch norm when True
		c_in	Number of input channels
		c_out	Number of output channels
		ks		Kernel size (size of the convolving kernel)
		sd		Stride of the convolution. Default: 1
	"""
	
	if bn:
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, bias=False),
			nn.BatchNorm2d(c_out),
			nn.ReLU(),
		)
	else:
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, bias=True),
			nn.ReLU(),
		)


def fc(f_in, f_out, activation=None, dropout=None):
	"""
	Args:
		f_in			Number of input feature
		f_out			Number of output feature
		activation		Choose from ['relu', 'sigmoid', 'logsoftmax', 'None'], default None
	"""
	
	if dropout is not None:
		if activation == 'relu':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				torch.nn.Dropout(dropout),
				nn.ReLU(),
			)
		elif activation == 'sigmoid':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				nn.Dropout(dropout),
				nn.Sigmoid(),
			)
		elif activation == 'logsoftmax':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				torch.nn.Dropout(dropout),
				nn.LogSoftmax(),
			)
		else:
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				torch.nn.Dropout(dropout),
			)
	else:
		if activation == 'relu':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				nn.ReLU(),
			)
		elif activation == 'sigmoid':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				nn.Sigmoid(),
			)
		elif activation == 'logsoftmax':
			return nn.Sequential(
				nn.Linear(f_in, f_out),
				nn.LogSoftmax(),
			)
		else:
			return nn.Sequential(  # bug吗，必须统一用sequential，即使只有一层
				nn.Linear(f_in, f_out),
			)


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0)
	# nn.init.xavier_normal_(m.bias.data)
	elif isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.)

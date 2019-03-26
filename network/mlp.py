#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2019-03-17 14:40:02
Program: 
Description:

# This is an implementation of the forward and backward process of neural networks (3 layers)

"""

import numpy as np


def softmax(z, axis):
	# calculate softmax along axis. Firstly, x_i = x_i - max(x). Then, exp(x_i) / sum_j(exp(x_j))
	max_z = np.expand_dims(np.max(z, axis), axis)
	exp_z = np.exp(z - max_z)
	return exp_z / np.expand_dims(np.sum(exp_z, axis), axis)


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
	return sigmoid(z) * (1-sigmoid(z))


def tanh_grad(z):
	return 1 - z*z


def ce(y, y_):
	return np.sum(-y * np.log(y_))


class NeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
	
		self.W1 = np.random.normal(size=(input_size, hidden_size))  # 采用w左乘的形式
		self.b1 = np.random.normal(size=hidden_size)
		self.W2 = np.random.normal(size=(hidden_size, output_size))
		self.b2 = np.random.normal(size=output_size)
		
		self.W1_grad = np.zeros(shape=(input_size, hidden_size))
		self.b1_grad = np.zeros(shape=hidden_size)
		self.W2_grad = np.zeros(shape=(hidden_size, output_size))
		self.b2_grad = np.zeros(shape=output_size)
		
		self.x = None
		self.hidden = None
		
	def forward(self, x):
		# x: bs x input_size
		self.x = x
		self.hidden = sigmoid(np.matmul(x, self.W1) + self.b1)  # bs x hidden_size
		output = np.matmul(self.hidden, self.W2) + self.b2  # bs x output_size
		return softmax(output, axis=1)

	def backward(self, y, y_):
		# y: bs x output_size
		# y_: bs x output_size，softmax后的输出分布
		
		# 1. calculate gradient of z^2 (也即ce误差函数对output的偏导)
		delta_output = y_ - y  # bs x output_size
		# calculate gradient of b^2（考虑到输入为一个batch，需要对b以及w的梯度在batch上累加）
		self.b2_grad = np.sum(delta_output, axis=0)  # output_size
		# calculate gradient of W^2
		self.W2_grad = np.matmul(self.hidden.T, delta_output)  # hidden_size x output_size
		
		# 2. calculate gradient of z^1
		delta_z = np.multiply(np.matmul(delta_output, self.W2.T), sigmoid_prime(self.hidden))  # bs x hidden_size
		# calculate gradient of b^1
		self.b1_grad = np.sum(delta_z, axis=0)  # hidden_size
		# calculate gradient of W^1
		self.W1_grad = np.matmul(self.x.T, delta_z)  # input_size x hidden_size
	
	def zero_grad(self):
		self.W1_grad = np.zeros(shape=(self.input_size, self.hidden_size))
		self.b1_grad = np.zeros(shape=self.hidden_size)
		self.W2_grad = np.zeros(shape=(self.hidden_size, self.output_size))
		self.b2_grad = np.zeros(shape=self.output_size)


class SGD:
	def __init__(self, lr=0.01):
		self.lr = lr

	def step(self, model):
		model.W2 -= self.lr * model.W2_grad
		model.b2 -= self.lr * model.b2_grad
		model.W1 -= self.lr * model.W1_grad
		model.b1 -= self.lr * model.b1_grad


model = NeuralNetwork(8, 3, 8)
sgd = SGD(lr=0.1)

x = np.eye(8)
for i in range(10000):
	y_ = model.forward(x)
	loss = ce(x, y_)
	if i % 100 == 0:
		print("epoch {}, loss: {}".format(i, loss))
	model.backward(x, y_)
	sgd.step(model)
	model.zero_grad()

y_ = model.forward(x)
np.set_printoptions(precision=1, suppress=True)
print('input:')
print(x)
print("hidden:")
print(model.hidden)
print('output:')
print(y_)

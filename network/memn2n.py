#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2019-02-25 09:36:12
Program: 
Description:

This is an implementation of MemN2N (End-To-End Memory Networks)

Reference

- https://github.com/liufly/memn2n

"""

import torch
import numpy as np
import torch.nn as nn
from tools.misc import to_var, get_parameter_number


def position_encoding(sentence_size, embedding_dim):
	encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
	ls = sentence_size + 1
	le = embedding_dim + 1
	for i in range(1, le):
		for j in range(1, ls):
			encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
	encoding = 1 + 4 * encoding / embedding_dim / sentence_size
	# Make position encoding of time words identity to avoid modifying them
	encoding[:, -1] = 1.0
	return np.transpose(encoding)


class MemN2N(nn.Module):
	"""Main class of MemN2N

	:param vocab_size: vocabulary size
	:param memory_size: max memory size
	:param sent_size: max sentence size
	:param embed_size: embedding size of both A and W
	:param max_hops: max hop number
	:param candidates: total candidates sets, n x sent_size
	
	Input shape:
		story: bs x memory_size x sent_size
		query: bs x sent_size
	
	Output shape:
		logits: bs x n
	
	"""
	
	def __init__(self, vocab_size, memory_size, sent_size, embed_size, max_hops, candidates):
		super(MemN2N, self).__init__()
		
		self.vocab_size = vocab_size
		self.memory_size = memory_size
		self.sent_size = sent_size
		self.embed_size = embed_size
		self.max_hops = max_hops
		self.candidates = candidates
		
		self.A = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.A.weight.data.normal_(0, 0.1)
		# norms = torch.norm(self.A.weight, p=2, dim=1).data
		# self.A.weight.data = self.A.weight.data.div(norms.view(vocab_size, 1).expand_as(self.A.weight))
		
		self.W = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.W.weight.data.normal_(0, 0.1)
		# norms = torch.norm(self.W.weight, p=2, dim=1).data
		# self.W.weight.data = self.W.weight.data.div(norms.view(vocab_size, 1).expand_as(self.W.weight))
		
		self.H = nn.Linear(self.embed_size, self.embed_size, bias=False)
		self.softmax = nn.Softmax(dim=1)
	
	def forward(self, story, query):
		story_size = story.size()  # bs x m x s
		
		u = list()
		query_embed = self.A(query)  # bs x s x d
		
		u.append(torch.sum(query_embed, 1))  # [bs x d]
		
		for hop in range(self.max_hops):
			# embed memory
			m_embed = self.A(story.view(story.size(0), -1))  # bs x m x s -> bs x (m x s) -> bs x (m x s) x d
			m_embed = m_embed.view(story_size + (m_embed.size(-1),))  # bs x (m x s) x d --> bs x m x s x d
			m = torch.sum(m_embed, 2)  # bs x m x d
			
			# calculate probability
			u_temp = u[-1].unsqueeze(1).expand_as(m)  # bs x d -> bs x 1 x d -> bs x m x d 沿着通道1复制了m次
			prob = self.softmax(torch.sum(m * u_temp, 2))  # bs x m
			
			# compute returned vector
			prob_tmp = prob.unsqueeze(2).expand_as(m)  # bs x m -> bs x m x 1 -> bs x m x d
			o_k = torch.sum(m * prob_tmp, 1)  # bs x d
			
			# update u
			u_k = self.H(u[-1]) + o_k
			# u_k = u[-1] + o_k
			
			u.append(u_k)
		
		# u_final = self.H(u[-1])
		
		# calculate similarity between the last u and candidates
		candidates_embed = self.W(self.candidates)  # n x s x d
		candidates_embed = torch.sum(candidates_embed, 1)  # n x d
		return torch.matmul(u[-1], torch.transpose(candidates_embed, 0, 1))  # bs x n


if __name__ == '__main__':
	def verify_model():
		vocab_size = 50
		memory_size = 10
		sent_size = 5
		embed_size = 8
		candidates = to_var(np.random.randint(0, vocab_size, (100, sent_size)))
		model = MemN2N(vocab_size=vocab_size,
		               memory_size=memory_size,
		               sent_size=sent_size,
		               embed_size=embed_size,
		               max_hops=3,
		               candidates=candidates)
		print(model)
		print('Parameter Numbers: {:,d}'.format(get_parameter_number(model)))
		
		if torch.cuda.is_available():
			model = nn.DataParallel(model.cuda(), device_ids=[0], dim=0)
		
		story = to_var(np.random.randint(0, vocab_size, (32, memory_size, sent_size)))
		query = to_var(np.random.randint(0, vocab_size, (32, sent_size)))
		answer = to_var(np.random.randint(0, 100, (32,)))
		
		model.train()
		similarity = model(story, query)
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		
		loss = criterion(similarity, answer)
		optimizer.zero_grad()  # clear gradients for this training step
		loss.backward()  # bp, compute gradients
		optimizer.step()
		
		print(similarity.size())
	
	verify_model()

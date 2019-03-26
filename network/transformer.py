#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2019-02-24 22:31:34
Program: 
Description: 

This is an implementation of Transformer (Attention Is All You Need)

Reference

- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html

"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tools.misc import to_var, get_parameter_number


PAD_token = 0
UNK_token = 1
EOS_token = 2
SOS_token = 3
EXCEPT = [PAD_token, UNK_token, EOS_token, SOS_token]


######################################################################
# TODO: 1. Scaled Dot Product Attention
######################################################################
class ScaledDotProductAttention(nn.Module):
	"""Scaled dot product attention, attention = softmax(Q * K^T / sqrt(d_k)) * V
	
	- bs = n_head x bs，并行计算多头
	- d_q = d_k，保证可以计算 Q * K^T
	- len_k = len_v，保证可以计算 softmax() * V
	
	Input shape:
		q: bs x len_q x d_k
		k: bs x len_k x d_k
		v: bs x len_v x d_v
		mask: bs x len_q x len_k
	
	Output shape:
		output: bs x len_q x d_v
		attn: bs x len_q x len_k
	"""
	
	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)
		self.softmax = nn.Softmax(dim=2)
	
	def forward(self, q, k, v, mask=None):
		attn = torch.bmm(q, k.transpose(1, 2))  # bs x len_q x d_k (bmm) bs x d_k x len_k --> bs x len_q x len_k
		attn = attn / self.temperature
		
		# 计算query中每个单词对key中所有单词的点积，故key中原来是pad的位置的logits需要置为0，也即只需softmax之前置为-inf即可
		if mask is not None:
			attn = attn.masked_fill(mask, -np.inf)
		
		attn = self.softmax(attn)
		attn = self.dropout(attn)
		output = torch.bmm(attn, v)  # bs x len_q x len_k (bmm) bs x len_v x d_v --> bs x len_q x d_v
		
		return output, attn


######################################################################
# TODO: 2. Multi-head Attention and Feed Forward
######################################################################
class MultiHeadAttention(nn.Module):
	"""Multi-head attention layer + Add layer + Norm layer
	
	Input shape:
		q: bs x len_q x d_model
		k: bs x len_k x d_model
		v: bs x len_v x d_model
		mask: bs x len_q x len_k
	
	Output shape:
		output: bs x len_q x d_model (self-attention vector for each time step)
		attn: (n_head x bs) x len_q x len_k
	"""
	def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
		super().__init__()
		
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		self.w_qs = nn.Linear(d_model, n_head * d_k)
		self.w_ks = nn.Linear(d_model, n_head * d_k)
		self.w_vs = nn.Linear(d_model, n_head * d_v)
		nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
		
		self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
		self.layer_norm = nn.LayerNorm(d_model)
		
		self.fc = nn.Linear(n_head * d_v, d_model)
		nn.init.xavier_normal_(self.fc.weight)
		
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, q, k, v, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		
		sz_b, len_q, _ = q.size()
		sz_b, len_k, _ = k.size()
		sz_b, len_v, _ = v.size()
		
		residual = q
		
		# 对n_head个多头同时线性变换，linear只会对输入的最后一个维度进行变换
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # bs x len_q x d_word_vec --> bs x len_q x n_head x d_k
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		
		# 把n_head和bs放到一个维度上，相当于扩大了bs，便于计算
		q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n_head x bs) x len_q x d_k
		k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n_head x bs) x len_k x d_k
		v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n_head x bs) x len_v x d_v
		
		mask = mask.repeat(n_head, 1, 1)  # bs x len_q x len_k --> (n_head x bs) x len_q x len_k
		output, attn = self.attention(q, k, v, mask=mask)  # (n_head x bs) x len_q x d_v, bs x len_q x len_k
		
		output = output.view(n_head, sz_b, len_q, d_v)
		output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # bs x len_q x (n_head x d_v) concat head
		
		output = self.dropout(self.fc(output))  # bs x len_q x d_model 再进行一次线性变换
		output = self.layer_norm(output + residual)
		
		return output, attn


class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise (in_channels: 512, out_channels: 2048)
		self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
		self.layer_norm = nn.LayerNorm(d_in)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		residual = x  # bs x len_q x d_model
		output = x.transpose(1, 2)  # bs x d_model x len_q (follow shape: bs x c x l)
		output = self.w_2(F.relu(self.w_1(output)))  # bs x d_model x len_q -> bs x d_hidden x len_q -> bs x d_model x len_q
		output = output.transpose(1, 2)  # bs x len_q x d_model
		output = self.dropout(output)
		output = self.layer_norm(output + residual)
		return output


######################################################################
# TODO: 3. single layer of encoder and decoder
######################################################################
class EncoderLayer(nn.Module):
	"""Single layer of encoder, attention layer + ffn layer
	
	the q, k, v of multi-head attention are all enc_input

	Input shape:
		enc_input: bs x input_len x d_model
		non_pad_mask: bs x input_len x 1，序列中非pad位置的值为1，pad位置都为0
		slf_attn_mask: bs x input_len x input_len, 序列中pad位置的值为true，非pad为false
	
	Output shape:
		enc_output: bs x input_len x d_model
		enc_slf_attn: (n_head x bs) x input_len x input_len
	"""
	
	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(
			n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
	
	def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):
		# 1. Multi-Head Attention. 输出和输入的维度一样，每个单词的位置都有一个编码
		enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
		# mask的作用：把输入序列中pad的位置的输出编码置为零
		enc_output *= non_pad_mask
		
		# 2. Feed Forward
		enc_output = self.pos_ffn(enc_output)
		enc_output *= non_pad_mask
		
		return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
	"""Single layer of decoder, mask attention layer + attention layer + ffn layer
	
	Input shape:
		dec_input: bs x output_len-1 x d_model
		enc_output: bs x input_len x d_model
		non_pad_mask: bs x output_len-1 x 1
		slf_attn_mask: bs x output_len-1 x output_len-1  训练时候知道tgt长度，所以可以这样操作；测试时候必须单步操作！
		dec_enc_attn_mask: bs x output_len-1 x input_len
	
	Output shape:
		dec_output: bs x output_len-1 x d_model
		dec_slf_attn: (n_head x bs) x output_len-1 x output_len-1
		dec_enc_attn: (n_head x bs) x output_len-1 x input_len
	"""
	
	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(DecoderLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
	
	def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
		# 1. Masked Multi-Head Attention, dec_output: bs x output_len-1 x d_model
		dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
		dec_output *= non_pad_mask
		
		# 2. Multi-Head Attention, dec_output: bs x output_len-1 x d_model
		dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
		dec_output *= non_pad_mask
		
		# 3. Feed Forward
		dec_output = self.pos_ffn(dec_output)
		dec_output *= non_pad_mask
		
		return dec_output, dec_slf_attn, dec_enc_attn


######################################################################
# TODO: 4. Multi layer of encoder and decoder
######################################################################
def get_non_pad_mask(seq):
	return seq.ne(PAD_token).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	""" Sinusoid position encoding table, 'i'取值为[1, d_model // 2]

	position    pos_enc
	0(PAD)      zero vec
	1           cos
	2           sin
	"""
	
	def cal_angle(position, hid_idx):
		return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)  # 0 0 2 2 ... 510 510 (偶数位置还是i，奇数位置i-1)
	
	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
	
	sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])  # n x 512
	
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
	
	if padding_idx is not None:
		# zero vector for padding dimension
		sinusoid_table[padding_idx] = 0.
	
	return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
	"""For masking out the padding part of key sequence

	:param seq_k: bs x len_k
	:param seq_q: bs x len_q
	"""
	
	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(PAD_token)  # bs x len_k，padded position in seq_k is True
	padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # bs x len_q x len_k，copy padding_mask len_q times
	
	return padding_mask


def get_subsequent_mask(seq):
	""" For masking out the subsequent info """
	
	sz_b, len_s = seq.size()
	subsequent_mask = torch.triu(
		torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # bs x len_seq x len_seq
	
	return subsequent_mask


class Encoder(nn.Module):
	"""Encoder network: layer_num x (multi-head attention + feed forward)
	输入source序列
	输出source序列中每个位置的单词的编码，以及每层网络的attn
	
	Input shape:
		src_seq: bs x input_len
		src_pos: bs x input_len
	Output_shape:
		enc_output: bs x input_len x d_model
		enc_slf_attn_list:
	"""
	
	def __init__(self, n_src_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()
		
		n_position = len_max_seq + 1
		
		self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD_token)
		
		self.position_enc = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
			freeze=True)  # 53 x d_model, clever! get the encoder vector by position
		
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
	
	def forward(self, src_seq, src_pos, return_attns=False):
		enc_slf_attn_list = list()
		
		# 1. Prepare masks
		# 1.1 标记key中padded的位置（bs x len_q x len_k），目的是消除这些位置对query的影响
		slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)  # bs x input_len x input_len
		# 1.2 标记query中没有pad的位置（bs x len_k x 1），query中原本为pad的vector，经过encoder后置为零
		non_pad_mask = get_non_pad_mask(src_seq)  # bs x input_len x 1
		
		# 2. Word embedding and position embedding
		enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)  # bs x input_len x d_word_vec
		
		# 3. Multi layer of encoder
		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(
				enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask)  # bs x input_len x d_model
			if return_attns:
				enc_slf_attn_list += [enc_slf_attn]
		
		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output,


class Decoder(nn.Module):
	def __init__(self, n_tgt_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()
		
		n_position = len_max_seq + 1
		
		self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD_token)
		self.position_enc = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
			freeze=True)
		
		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
	
	def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
		dec_slf_attn_list, dec_enc_attn_list = [], []
		
		# 1. Prepare masks
		# 1.1 把pad的地方都填零，非pad的地方填1，目的：消除pad的单词的位置对概率分布的影响
		non_pad_mask = get_non_pad_mask(tgt_seq)  # bs x output_len-1 x 1
		
		# 1.2 把当前单词后面的单词都标志为mask [[0 1 1 ... 1] [0 0 1 ... 1] ...]
		slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)  # bs x output_len-1 x output_len-1
		# 1.3 标记key中pad的位置，目的：消除这些位置对query的影响 （bs x len_q x len_k）
		slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)  # bs x output_len-1 x output_len-1
		# 融合1.2和1.3，只要为当前时刻后面的序列，或者PAD的，都要mask
		slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
		
		dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)  # 消除padded k对q的影响
		
		# 2. Word embedding and position embedding
		dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
		
		# 3. Multi layer of decoder
		for dec_layer in self.layer_stack:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask,
				dec_enc_attn_mask=dec_enc_attn_mask)
			
			if return_attns:
				dec_slf_attn_list += [dec_slf_attn]
				dec_enc_attn_list += [dec_enc_attn]
		
		if return_attns:
			return dec_output, dec_slf_attn_list, dec_enc_attn_list
		return dec_output,


######################################################################
# TODO: 5. Transformer
######################################################################
class Transformer(nn.Module):
	"""Main class of transformer
	
	:param n_src_vocab: source language's vocabulary size
	:param n_tgt_vocab: target language's vocabulary size
	:param len_max_seq: max sequence length between input sequence and output sequence
	:param d_word_vec: (512) word embedding size, == d_model
	:param d_model: (512) model dimension, == n_head * d_k
	:param d_inner: (2048) hidden size of feed forward network
	:param n_layers: (6), layer number of encoder and decoder
	:param n_head: (8), number of multi-head
	:param d_k: (64), dimension of key in a single head, == d_q
	:param d_v: (64), dimension of value in a single head
	:param dropout: (0.1)
	:param tgt_emb_prj_weight_sharing: share the same weight matrix between word embedding and pre-softmax linear
	:param emb_src_tgt_weight_sharing: share the same weight matrix between source and target word embedding
	
	Input shape:
		src_seq: bs x input_len, padded with zero-index
		src_pos: bs x input_len, count from 1, fill with 0 for padded tokens
		tgt_seq: bs x output_len, padded with zero-index
		tgt_pos: bs x output_len, count from 1, fill with 0 for padded tokens
	Output shape:
		logit_final: (bs x (output_len - 1)) x vocab_size
	"""
	
	def __init__(
			self,
			n_src_vocab, n_tgt_vocab, len_max_seq,
			d_word_vec=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
			tgt_emb_prj_weight_sharing=True,
			emb_src_tgt_weight_sharing=True):
		super().__init__()
		
		self.encoder = Encoder(
			n_src_vocab=n_src_vocab, len_max_seq=len_max_seq, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
		
		self.decoder = Decoder(
			n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
		
		self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
		nn.init.xavier_normal_(self.tgt_word_prj.weight)
		
		assert d_model == d_word_vec, \
			'To facilitate the residual connections, the dimensions of all module outputs shall be the same.'
		
		if tgt_emb_prj_weight_sharing:
			# Share the weight matrix between target word embedding and the final logit dense layer
			self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
			self.x_logit_scale = (d_model ** -0.5)  # ref to: Using the output embedding to improve language models
		else:
			self.x_logit_scale = 1.
		
		if emb_src_tgt_weight_sharing:
			# Share the weight matrix between source and target word embeddings
			assert n_src_vocab == n_tgt_vocab, \
				"To share word embedding table, the vocabulary size of src/tgt shall be the same."
			self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight
	
	def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
		tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]  # bs x output_length-1, offset one position!
		
		enc_output, *_ = self.encoder(src_seq, src_pos)  # bs x input_len x d_model
		dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)  # bs x output_len-1 x d_model
		logit_seq = self.tgt_word_prj(dec_output) * self.x_logit_scale  # bs x output_len-1 x vocab_size
		
		logit_final = logit_seq.view(-1, logit_seq.size(2))  # (bs x (output_len - 1)) x vocab_size
		
		return F.log_softmax(logit_final, -1)


if __name__ == '__main__':
	
	def verify_model():
		vs = 1000
		input_len = 10
		output_len = 10
		max_len = max(input_len, output_len)
		
		model = Transformer(n_src_vocab=vs, n_tgt_vocab=vs, len_max_seq=max_len,
		                    d_word_vec=512, d_model=512, d_inner=2048,
		                    n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
		                    tgt_emb_prj_weight_sharing=True, emb_src_tgt_weight_sharing=True)
		print(model)
		print('Parameter Numbers: {:,d}'.format(get_parameter_number(model)))
		
		if torch.cuda.is_available():
			model = nn.DataParallel(model.cuda(), device_ids=[0], dim=0)

		# prepare input data
		src_seq = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
		           [3, 2, 5, 3, 1, 6, 8, 3, 0, 0]]
		
		src_pos = np.array([
			[pos_i + 1 if w_i != PAD_token else 0 for pos_i, w_i in enumerate(inst)] for inst in src_seq])
		
		tgt_seq = [[3, 2, 5, 3, 1, 6, 8, 3, 0, 0],
		           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
		tgt_pos = np.array([
			[pos_i + 1 if w_i != PAD_token else 0 for pos_i, w_i in enumerate(inst)] for inst in tgt_seq])
		
		src_seq = to_var(np.array(src_seq))
		src_pos = to_var(src_pos)
		tgt_seq = to_var(np.array(tgt_seq))
		tgt_pos = to_var(tgt_pos)
		
		output = model(src_seq, src_pos, tgt_seq, tgt_pos)
		print(output.size())
	
	verify_model()

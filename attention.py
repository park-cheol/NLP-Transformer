""" Attention 모듈을 따로 모아놓은 python file
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class SelfAttention(nn.Module):

    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.attention_dim = args.hidden_dim // args.n_head

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)

        # 초기화 Linear
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(args.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).cuda(args.gpu)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch_size, sentence_length, hidden_dim]
        # paper) mask는 decoder의 auto regression propety를 위해서 사용할 예정

        # Q, K, V 행렬 생성
        # 동일한 input sentence 사용하고 self-attention score 계산을 위해
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k ,v = [batch_size, sentence_length, attention_dim]

        # paper) Attention = softmax(QK^T / sqrt(d_k)) * V
        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch_size, sentence_length, sentence_length]

        if mask is not None:
            # todo print mask 어떻게 생겼는지
            self_attention = self_attention.masked_fill(mask, -np.inf)
            # a.masked_fill(mask, value): a에 mask 값이 True 일 시 value값으로 대체
            # a = [0.5 0.1 -0.3] ==> a.maksed_fill(a > 0, -inf)
            # a = [-inf, -inf, -0.3]

        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch_size, sentence_length, sentence_length]

        weighted_v = torch.bmm(norm_attention_score, v)
        # [batch_size, sentence_length, attention_dim]
        # bmm => 내적곱이라고 생각 하면됨 (seq_length x seq_length)*(seq_length x attention_dim)
        # ==> (seq_length, attention_dim)

        return self.dropout(weighted_v), attention_score


class MultiHeadAttention(nn.Module):

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        assert args.hidden_dim % args.n_head == 0 # embed_dim 은 head 수만큼 나눌 예정이므로
        self.attentions = nn.ModuleList([SelfAttention(args)
                                         for _ in range(args.n_head)]) # h_head만큼 실행

        self.o_w = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        init_weight(self.o_w)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query, key, value, mask=None):
        # q k ,v = [batch_size, sentence_length, hidden_dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions [batch_size, sentence_length, attention_dim] * n_head

        weight_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        weighted_v = torch.cat(weight_vs, dim=-1)
        # [batch_size, sentence_length, hidden_dim]

        output = self.dropout(self.o_w(weighted_v))
        # [batch_size, sentence_length, hidden_dim]

        return output, attentions



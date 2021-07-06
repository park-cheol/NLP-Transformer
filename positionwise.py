""" PositionWise
encoder/decoder 에서 Multi-head attention 이후에 해 줄 Feedforward 부분 따로 선언
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# todo FFN이 해주는 역할
class PositionWiseFeedForward(nn.Module):

    def __init__(self, args):
        super(PositionWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(args.hidden_dim, args.feed_forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(args.feed_forward_dim, args.hidden_dim, kernel_size=1)
        # nn.Conv1d 의 input (N, C) => N: batch_size C: Channels

        init_weight(self.conv1)
        init_weight(self.conv2)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        # input [batch_size, sentence_length, hidden_dim]

        # input 을 conv input 형태로 바꿔줌
        input = input.permute(0, 2, 1) # [batch_size, hidden_dim, sentence_length]
        conv1 = self.conv1(input) # [batch_size, feed_forward_dim, sentence_length]
        conv1_relu = F.relu(conv1)
        conv1_relu_drop = self.dropout(conv1_relu)

        conv2 = self.conv2(conv1_relu_drop) # [batch_size, hidden_dim, sentence_length]

        output = conv2.permute(0, 2, 1) # [batch_size, sentence_length, hidden_dim]

        return self.dropout(output)



























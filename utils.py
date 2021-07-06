import os
import re
import json
import pickle
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext import data as ttd
from torchtext.data import Example, Dataset

def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

########################
# Scheduling Adam
########################
# paper) 초기에 lr을 Linear 하게 증가시키고 후반에 inverse square root 로 감소
class ScheduleAdam():

    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5) # 1 / sqrt(hidden_dim)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # current_step 정보를 이용해서 lr Update
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr # lr Update

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5),
                       self.current_steps * np.power(self.warm_steps, -1.5)
                       ])


################################
# attention plot
def display_attention(candidate, translation, attention):
    """
    displays the model's attention over the source sentence for each target token generated.
    Args:
        candidate: (list) tokenized source tokens
        translation: (list) predicted target translation tokens
        attention: a tensor containing attentions scores
    Returns:
    """
    # attention = [target length, source length]

    attention = attention.cpu().detach().numpy()
    # attention = [target length, source length]

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(fname=font_location)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

























import os
import re
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext import data
from torchtext.data import Example, Dataset


#########################
# convert to dataset
#########################
def clean_text(text):
    """
    normalize 한 입력 sentence 에서 특수 문자 제거
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text

def convert_to_dataset(data, kor, eng):
    """
    pandas 를 torchtext dataset 으로 변환
    """

    # str 이 아닌 것을 없애버림
    missing_rows = [idx for idx, row in data.iterrows() if type(row.korean) != str or type(row.english) != str]
    data = data.drop(missing_rows)

    # 변환하기
    list_of_examples = [Example.fromlist(row.apply(lambda x: clean_text(x)).tolist(),
                                         fields=[('kor', kor), ('eng', eng)]) for _, row in data.iterrows()]
    # todo Example ?? 안에 어떻게 내용이 들어가 있는지 확인?

    dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])

    return dataset

###########################
# Load Dataset
###########################
def load_dataset(mode):
    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        train_file = os.path.join(data_dir, 'train.csv')
        train_data = pd.read_csv(train_file, encoding='utf-8')

        valid_file = os.path.join(data_dir, 'valid.csv')
        valid_data = pd.read_csv(valid_file, encoding='utf-8')

        print("train 총: ", len(train_data))
        print("valid 총: ", len(valid_data))

        return train_data, valid_data

    else:
        test_file = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print("test 총: ", len(test_data))

        return test_data

# todo 이부분은 나중에 자세히
def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    """
    pandas -> torchtext Dataset으로 바꾸고 iteration 만들어냄
    """
    file_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(file_kor)

    file_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(file_eng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # convert pandas DataFrame to torchtext dataset
    if mode == 'train':
        train_data = convert_to_dataset(train_data, kor, eng)
        valid_data = convert_to_dataset(valid_data, kor, eng)

        # make iterator using train and validation dataset
        print(f'Make Iterators for training . . .')
        train_iter, valid_iter = data.BucketIterator.splits(
            (train_data, valid_data),
            # the BucketIterator needs to be told what function it should use to group the data.
            # In our case, we sort dataset using text of example
            sort_key=lambda sent: len(sent.kor),
            # all of the tensors will be sorted by their length by below option
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return train_iter, valid_iter

    else:
        test_data = convert_to_dataset(test_data, kor, eng)

        # defines dummy list will be passed to the BucketIterator
        dummy = list()

        # make iterator using test dataset
        print(f'Make Iterators for testing . . .')
        test_iter, _ = data.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter









































































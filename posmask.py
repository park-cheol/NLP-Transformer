""" Position embedding 파일
RNN을 사용하지 않기에 순서의 정보를 줄 수 없으므로 따로 pos embedding하여 add해줌
"""
import numpy
import numpy as np
import pickle
import sys
"""일반 텍스트를 파일로 저장 시 파일 i/o 이용 하지만 list나 class 같은 자료형은 데이터를 저장하거나
불러올 수 없다.
이럴 때 pcikle 이라는 모듈 사용
원하는 데이터를 변경없이 파일로 저장하여 그대로 로드 할 수 있음
저장하거나 불러올때는 바이트형식 (wb, rb)
wb로 데이터 입력시 .bin확장자 사용 / 모든 파이썬 데이터 객체를 저장하고 읽을 수 있음

입력: pickle.dump(data, file)
로드: 변수 = pickle.load(file)
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize)
pickle_eng = open('pickles/eng.pickle', 'rb') # 읽기
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
# vocab.stoi 를 통해서 현재 단어 집합의 단어와 맵핑된 idx 출력 할 수있다


#####################
# Position Embed
#####################

def create_position_vector(sentence, args):
    print("----[Create_position_vector]----")
    # [batch_size, sentence_length]
    print("Sentence: ", sentence.size())
    # encoder [128, 8]
    # decoder [128, 22]
    batch_size, _ = sentence.size()
    pos_vec = np.array([(pos + 1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    print("pos_vec", pos_vec.shape)
    # encoder: (1024, )
    # decoder (2816, )
    # todo 다시 자세히 파악해보기 대략알겠음

    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).cuda(args.gpu)
    print("position_vector: ", pos_vec.size())
    # encoder [128, 8]
    # decoder [128, 22]

    return pos_vec

def create_position_encoding(max_len, hidden_dim, args):
    # max_len = 65, hidden_dim = 512
    print("----[Create_position_Encoding]----")
    # paper)
    # PE(pos, 2i) = sin(pos / 10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos / 10000 ** (2*i / hidden_dim))
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    print("sinusoid_table:", sinusoid_table.shape)
    # (65*512=33280, )
    # sinusoid_table = [max_len * hidden_dim]

    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    print("sinusoid_table(reshape): ", sinusoid_table.shape)
    # (65, 512) array
    # [max_len, hidden_dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # 짝수
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # 홀수
    # paper) 2i + 1을 이런식으로 표현
    # Tensor로 변환하고 batc size 배로 repeat
    sinusoid_table = torch.FloatTensor(sinusoid_table).cuda(args.gpu)
    sinusoid_table[0] = 0. # todo ???
    print("Position_Encoding: ", sinusoid_table.size())

    # [65, 512]
    return sinusoid_table

#########################
# Mask
#########################
def create_source_mask(source):
    """
    encoder self attention 에 쓰일 masking Tensor 생성
    만약 [[2, 193, 9, 27, 10003, 1, 1, 1, 3] 2 = <sos>, 3 = <eos> 1 = <pad>
    masking Tensor = [False, False, False, False, False, True, True, True, False]
    """
    # source [batch_size, source_length] = [128, 6]
    source_length = source.shape[1]

    # source 와 target sentence 의 padding tokens 만들기 위해서 사용되어질 bool tensor 생성
    source_mask = (source == pad_idx)
    # source 중에 pad_idx element 는 True
    # [batch_size, source_length]

    # sentence length 배만큼 repeat
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # [batch_size, source_length, source_length]

    return source_mask


def create_subsequent_mask(target, args):
    """
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    식으로 생성하고 bool을 붙혀서 0=False, 1=True 로 바꿈
    """
    # [batch_size, target_length]

    batch_size, target_length = target.size()

    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().cuda(args.gpu)
    # [target_length, target_length]
    # torch.triu(input, diagonal=0, *, out=None) 경우
    # [1 1 1 1 1]
    # [0 1 1 1 1]
    # [0 0 1 1 1]
    # [0 0 0 1 1]
    # [0 0 0 0 1]
    # upper triangular part 를 만듬 diagonal = 1 인 경우 (0,1)에서 시작
    # -1 인 경우 (1,0)에서 시작

    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # [batch_size, target_length, target_length]

    return subsequent_mask

def create_target_mask(source, target, args):
    """
    decoder self attention 과 encoder output을 이용한 decoder attention 할 때 사용할 mask
    똑같이 2 = <sos>, 3 = <eos>, 1 = <pad>
    """
    # source [batch_size, source_length]
    # target [batch_size, target_length]

    target_length = target.shape[1]

    subsequent_mask = create_subsequent_mask(target, args)
    # [batch_size, target_length, target_length]

    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # [batch_size, target_length]

    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    # source_mask 와 동일하지만 차원만 맞춰줌
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # combine
    target_mask = target_mask | subsequent_mask
    # [[[False, False, True .... ]]]같이 찍힘
    # | : 비트연산자 or 즉, 둘 중 하나 True 이면 1
    # [batch_size, target_length, target_length]

    return target_mask, dec_enc_mask

























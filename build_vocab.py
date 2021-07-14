
import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import *
from datasets import *

from torchtext import data
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
"""soynlp: 한국어 처리를 위한 파이썬 패키지 중 하나
여기서 제공하는 형태소분석기는 형태소기반으로 문서를 token화 할 수 있는 기능 제공 but 새롭게 만들어진
미등록 단어들은 인식이 잘 안됌
custom으로 사전에 단어를 등록해야하는 절차 걸처야 함
WordExtractor: 형태소에 해당하는 단어를 분리하는 학습을 수행
LToeknizer: 한국어의 경우 띄어쓰기로 분리된 하나의 문자열은 "L token + R token" 구조가 대다수
L: 명사, 대명사, 동사, 형용사 / R: 조사, 동사, 형용사
여러 길이의 L 토큰의 점수를 비교하여 가장 점수가 높은 L단어를 찾는것이 L-tokenizing
"""

# 한국어 input sentence로 쓰일 soynlp tokenizer를 학습
def build_tokenizer():
    print("토큰 진행 생성 중")

    data_dir = Path().cwd() / 'data'
    # cwd: 현재 폴더 경로
    # '/home/jnu2/python_CH/transformer/data'
    train_file = os.path.join(data_dir, 'corpus.csv')

    df = pd.read_csv(train_file, encoding='utf-8')
    # index와 함께 korean english reading

    # text가 아닌 것을 만나면 skip
    kor_lines = [row.korean for _, row in df.iterrows() if type(row.korean) == str]
    # korean row 만 봄

    word_extractor = WordExtractor(min_frequency=5)
    # 단어를 통계를 이용하여 만듬: Branching Entropy, Accessor Variety, Cohesion Score
    # 자세한것 갤탭에
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    # extract로 chhesion, branching entropy, accessor variety 등 통계 수치를 계산 가능
    # https://datascienceschool.net/03%20machine%20learning/03.01.04%20soynlp.html

    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()} # todo ??

    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)
        # pickle에 입력 저장

def build_vocab(args):
    """
    입력 sentence를 word indices로 변경으로 사용될 vocab 생성
    """

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    # 저장했던것 (chhesion) 불러옴
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)
    """
    Cohesion을 이용하여 어절 왼쪽에 위치한 substrings 중 가장 단어스러운 부분을 찾는 방식(비지도 학습기반 tokenzier)
    subword 빈도 P(AB|A) cohesion scroe
    아이     4천  0.15      0.15
    아이오   307  0.06      0.1
    아이오아  270  0.88      0.2
    아이오아이 270  1        0.3
    그럼 아이오아이가 가장 cohesion score가 높으므로 어절은 '아이오아이 + 는'으로 나눠짐
    """
    # Field: 앞으로의 전처리를 어떻게 할지를 결정
    # tokenize: 토큰화 함수를 어떻걸 사용할것인지

    kor = data.Field(tokenize=tokenizer.tokenize,
                     lower=True,
                     batch_first=True)

    # spacy는 한국어를 아직 지원X, 텍스트 전처리에 좋음
    # Field에 'spacy'이 내장되어있음
    # Field는 전처리를 정의하는 함수
    eng = data.Field(tokenize='spacy',
                     init_token='<sos>',
                     eos_token='<eos>',
                     lower=True,
                     batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv') # Train dataset 가져옴
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, kor, eng)
    # Field 적용 시킨 후 kor-eng paired 되게 Dataset에 넣음

    print("vocab 생성 중")

    # 단어 집합 생성 max_size: 단어집합의 최대 크기, min_freq: 단어 집합에 추가시 최소 등장 빈도 추가
    kor.build_vocab(train_data, max_size=args.kor_vocab) # default= 55000
    eng.build_vocab(train_data, max_size=args.eng_vocab) # defaulut = 30000
    print("A",kor.vocab)
    print("B", kor.vocab.stoi)
    # 총 단어집합 수
    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}') # 55002: 3개는 pad sos eos
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}') #

    # 가장 많이 등장하던것 출력
    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    with open('pickles/kor.pickle', 'wb') as kor_file: # kor.vocab 저장, 단어에 indexing 되어있는것
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file: # eng.vocab 저장, 단어에 indexing 되어있음
        pickle.dump(eng, eng_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--kor-vocab', type=int, default=55000)
    parser.add_argument('--eng-vocab', type=int, default=30000)

    args = parser.parse_args()

    # build_tokenizer()
    build_vocab(args)


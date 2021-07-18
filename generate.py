import numpy as np
import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from datasets import clean_text
from utils import display_attention
from models import Transformer


parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='restart epoch number')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--mode', default='train', type=str,
                    help='train Or test')
parser.add_argument('--warm-steps', default=4000, type=int)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--resume', type=str)
# Model opt
parser.add_argument('--input-dim', default=0, type=int)
parser.add_argument('--output-dim', default=0, type=int)
parser.add_argument('--hidden-dim', default=512, type=int)
parser.add_argument('--feed-forward-dim', default=2048, type=int)
parser.add_argument('--n-layer', default=6, type=int)
parser.add_argument('--n-head', default=8, type=int)
parser.add_argument('--max-len', default=64, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--sos-idx', type=int, default=0)
parser.add_argument('--eos-idx', type=int, default=0)
parser.add_argument('--pad-idx', type=int, default=0)

# distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# predict
parser.add_argument('--input', type=str)

def generate():
    args = parser.parse_args()
    input = clean_text(args.input) # 특수 문자 제거

    # load tonkenizer and Field
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx = eng.vocab.stoi['<eos>'] # 3

    args.input_dim = len(kor.vocab)
    args.output_dim = len(eng.vocab)

    args.sos_idx = eng.vocab.stoi['<sos>']
    args.eos_idx = eng.vocab.stoi['<eos>']
    args.pad_idx = eng.vocab.stoi['<pad>']

    # model and load
    model = Transformer(args)
    model.load_state_dict(torch.load(args.resume))
    model = model.cuda(args.gpu)
    model.eval()

    # Tensor로 변환하고 forward
    tokenized = tokenizer.tokenize(input)
    indexed = [kor.vocab.stoi[token] for token in tokenized]
    # print("tokenized", tokenized) = ['그녀는', '아직', '결혼', '하지', '않았다']
    # print("indexed", indexed) = [207, 186, 322, 100, 0]

    source = torch.LongTensor(indexed).unsqueeze(0).cuda(args.gpu) # [1, source_len], tensor변환
    target = torch.zeros(1, args.max_len).type_as(source.data) # [1, max_len]
    # [1, 64]: target [[0, 0, 0, 0...]] 아마 이제 여기다가 인덱스를 부여할듯

    encoder_output = model.encoder(source) # [1, source_length, 512]
    next_symbol = eng.vocab.stoi['<sos>']
    # print("Symbol", next_symbol) = 2
    for i in range(0, args.max_len):
        target[0][i] = next_symbol
        # print("target", target) [[2, 0 , 0 ...]] --> [[2, 112, 0, 0]] ... 식으로 채워나감
        # print("target size", target.size()) = [1, 64]
        decoder_output, _ = model.decoder(target, source, encoder_output)
        # print("decoder output",decoder_output)
        # print("decoder output size",decoder_output.size()) = [1, 64, 19545]
        # [1, target_length, output_dim]

        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1] # index만 불러옴
        # print("prob,size", prob.size()) = [64]
        next_word = prob.data[i]
        # print("next_word", next_word) = 112 / 2번쨰: 240 -> 683 ->
        next_symbol = next_word.item()
        # print("next_symbol", next_symbol) = tensor(112) -> 122

    # print(target) 출력 영어는 [she still married yet she is married]
    # tensor([[  2, 112, 240, 683, 295, 112,   9, 683,   3,   3,   3,   3,   3,   3,
    #            3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
    #            3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
    #            3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
    #            3,   3,   3,   3,   3,   3,   3,   3]], device='cuda:0')
    """Greedy Search
    각 target의 단어(index)를 하나씩 예측하면서 반복하여 만들어냄
    """

    # torch.where(condition, x, y)
    # condition 에 따라 X 또는 Y에서 선택한 요소의 텐서를 반환: True=x, False=y
    # 또는 위의 같은 경우에는 idx를 반환
    eos_idx = int(torch.where(target[0] == eos_idx)[0][0])
    # print(torch.where(target[0] == eos_idx)) = []
    # eos_idx 가 처음에는 3인대 target 중에 eos_idx가 처음으로 나오는 곳을 찾아줌
    # 결국 eos_idx = 8
    target = target[0][:eos_idx].unsqueeze(0)
    # print(target): 위에서 저런게 [1,64]인것을 잘라줌
    # tensor([[  2, 112, 240, 683, 295, 112,   9, 683]], device='cuda:0')

    # 번역 tensor = [target length] <= word indice로 채워져있음

    # print(source.size()) = [1, 5]
    # print(target.size()) = [1, 8]
    target, attention_map = model(source, target)
    # print(target.size()) = [1, 8, 19545]
    # print(np.array(attention_map).shape) = (8, )
    # print(attention_map[0].size(), attention_map[1].size()) = 전부다 [1, 8, 5]
    # 즉 8=target length, 5= source length 이니까 source target의 연관성

    target = target.squeeze(0).max(dim=-1)[1]
    # print(target) = tensor([112, 240, 683, 295, 112,   9, 683,   3], device='cuda:0')
    # target 들 index

    translated_token = [eng.vocab.itos[token] for token in target] # token 불러오기
    # print(translated_token):['she', 'still', 'married', 'yet', 'she', 'is', 'married', '<eos>']
    translation = translated_token[:translated_token.index('<eos>')]
    # <eos> 전까지 출력
    # print(translation): ['she', 'still', 'married', 'yet', 'she', 'is', 'married']
    translation = ' '.join(translation)
    # print(translation): she still married yet she is married

    print(f'kor > {args.input}')
    print(f'eng > {translation.capitalize()}') # capitalize: 첫 앞문자만 대문자로
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])

if __name__ == '__main__':
    generate()




















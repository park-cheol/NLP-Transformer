import argparse
import datetime
import os
import random
import shutil
import time
import warnings
import pickle
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from datasets import *
from models import *
from utils import *

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
parser.add_argument('--resume', type=str, default=None)

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

def main():
    args = parser.parse_args()

    # Args update
    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)

    args.input_dim = len(kor.vocab) # 55002
    args.output_dim = len(eng.vocab) # 19545

    # stoi로 현재 단어에 맵핑된 고유 inx를 알 수 있음
    args.sos_idx = eng.vocab.stoi['<sos>'] # 2
    args.eos_idx = eng.vocab.stoi['<eos>'] # 3
    args.pad_idx = eng.vocab.stoi['<pad>'] # 1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    model = Transformer(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()


    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    else:
        print("No load state dict")
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx).cuda(args.gpu)
    optimizer = ScheduleAdam(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        hidden_dim=args.hidden_dim,
        warm_steps=args.warm_steps
    )

    cudnn.benchmark = True

    # Load Dataset and train
    if args.mode == 'train':
        train_data, valid_data = load_dataset(args.mode)
        train_iter, valid_iter = make_iter(args, args.batch_size, args.mode,
                                           train_data=train_data, valid_data=valid_data)
        train(args, model, train_iter, valid_iter, optimizer, criterion)
    else:
        test_data = load_dataset(args.mode)
        test_iter = make_iter(args, args.batch_size, args.mode, test_data=test_data)

        inference(args, model, test_iter, criterion)

def train(args, model, train_iter, valid_iter, optimizer, criterion):
    print("Parameter 수: ", model.count_params()) # 82,253,312
    best_valid_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        accuracy = 0
        iter = 0
        model.train()
        epoch_loss = 0
        start = time.time()

        for batch in train_iter:
            # print("----[Train]----")
            optimizer.zero_grad()
            source = batch.kor.cuda(args.gpu)
            # print("source: ", source.size())
            # source: [batch=128, source_length=8]

            target = batch.eng.cuda(args.gpu)
            # print("target: ", target.size())
            # target: [batch=128, target_length=23]

            # target sentences 는 <sos> 포함 (<eos>제외)
            output = model(source, target[:, :-1])[0]
            # todo print
            # GT sentence 는 <eos>는 포함 (<sos>는 제외)
            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            # todo print shape
            # output [(batch * target length -1), output_dim]
            # target [(batch * target length -1)]

            loss = criterion(output, target)
            loss.backward()

            # gradint exploding으로부터 방지하기 위해 사용
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

            optimizer.step()

            if iter % 500 == 0:
                print(f"Epoch: {epoch+1:02} | Iter{iter} / {len(train_iter)} | loss {loss:.3f}")

            iter += 1
            epoch_loss += loss.item()


        train_loss = epoch_loss / len(train_iter)
        valid_loss, accuracy = evaluate(args, model, valid_iter, criterion)
        accuracy = accuracy / len(valid_iter)

        if valid_loss < best_valid_loss: # loss 가 더 낮게 뜨면 저장
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "saved_models/model_%d.pth" % (epoch + 1))

        print(f"Epoch: {epoch+1:02} | Epoch Time: ", datetime.timedelta(seconds=time.time() - start))
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Acc: {accuracy:.3f}")



def evaluate(args, model, valid_iter, criterion):
    model.eval()
    epoch_loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch in valid_iter:
            source = batch.kor
            target = batch.eng

            output = model(source, target[:, :-1])[0]

            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:, 1:].contiguous().view(-1)

            accuracy += get_accuracy(output, target, args.pad_idx)
            loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(valid_iter), accuracy

def inference(args, model, test_iter, criterion):
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    BLEU_score = 0
    epoch_loss = 0
    i = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in test_iter:
            i += 1

            source = batch.kor
            target = batch.eng

            output = model(source, target[:, :-1])[0]

            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:, 1:].contiguous().view(-1)

            loss = criterion(output, target)

            accuracy = get_accuracy(output, target, args.pad_idx)
            total_accuracy += accuracy

            translated_token = [eng.vocab.itos[token] for token in target]
            translation = translated_token[:translated_token.index('<eos>')]
            translation_ = ' '.join(translation)

            pred = output.squeeze(0).max(dim=-1)[1]
            translated_token_pred = [eng.vocab.itos[token] for token in pred]
            translation_pred = translated_token_pred[:translated_token_pred.index('<eos>')]
            translation_pred_ = ' '.join(translation_pred)

            references = [translation]
            BLEU = sentence_bleu(references, translation_pred, weights=[1])

            BLEU_score += BLEU
            print("pred: ", translation_pred_)
            print("target ", translation_)
            epoch_loss += loss.item()
            print(i, "---------------------------------------------------------")

    BLEU_score = BLEU_score / len(test_iter)
    test_loss = epoch_loss / len(test_iter)
    total_accuracy / len(test_iter)
    print(f"BLEU score: {BLEU_score:.3f}")
    print(f'accuracy: {total_accuracy:.3f} %')
    print(f'Test loss: {test_loss:.3f}')






if __name__ == '__main__':
    main()
















































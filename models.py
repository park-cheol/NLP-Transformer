import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *
from utils import *
from positionwise import *
from posmask import *

class EncoderLayer(nn.Module):

    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(args)
        self.position_wise_ffn = PositionWiseFeedForward(args)

    def forward(self, source, source_mask):
        # print("----[Encoder_layer] * (default:6번)----")
        # print("Encoder_source: ", source.size())
        # [128, 8, 512] : pos_emb + token_emb 한 것
        # source [batch_size, source_length, hidden_dim]
        # source_mask [batch_size, source_length, source_length]
        # 여기서 source 는 embedding 후의 Tensor

        # paper 에서는 LaynorNorm(x + subLayer(x)) 이지만
        # Update: x + subLayer(LayerNorm(x))로 변경
        # todo 더 좋아진점
        normalized_source = self.layer_norm(source)
        # print("norm_source: ", normalized_source.size())
        # [128, 8, 512]
        self_attention = source + self.self_attention(
            normalized_source, normalized_source, normalized_source, source_mask)[0]
        # self_attention (Q, K, V, mask)
        # print("Encoder_Attention output:", self_attention.size())
        # [128, 8, 512]
        normalized_self_attention = self.layer_norm(self_attention)
        # print("norm_Attention: ", normalized_self_attention.size())
        output = self_attention + self.position_wise_ffn(normalized_self_attention)
        # output [batch_size, source_length, hidden_dim]
        # 참고) n_head 반복해야하므로 input = output 이다

        return output

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args
        # input_dim = 55002 (kor.vocab) / output_dim = 19545 (eng.vocab) / hidden_dim = 512(default)
        # pad_idx = 3
        self.token_embedding = nn.Embedding(args.input_dim, args.hidden_dim,
                                            padding_idx=args.pad_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=args.hidden_dim**-0.5)

        self.embedding_scale = args.hidden_dim ** 0.5
        # max_len = 64 (default)
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_position_encoding(args.max_len + 1, args.hidden_dim, args), freeze=True
        )
        # from_pretrained(embeddings, freeze=True, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        # freeze=True: 말 그대로 update를 training때 하지마라
        # 미리 학습된 vector들을 가져와 사용

        # n_layer = 6 (Default)
        self.encoder_layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)

    def forward(self, source):
        # print("----[Encoder]----")
        # source [batch_size, source_length] = [128, 6]
        # print("Encoder Source: ", source.size())
        # [128, 8]
        source_mask = create_source_mask(source) # [batch_size, source_length, source_length]
        # print("Source_mask: ", source_mask.size())
        # [128, 8 ,8]
        source_pos = create_position_vector(source, self.args) # [batch_size, source_length]
        # print("Source_position: ", source_pos.size())
        # [128, 8]

        embedding_scaled = self.token_embedding(source) * self.embedding_scale
        # print("token_embedding: ", embedding_scaled.size())
        # print("A", embedding_scaled)
        # [128, 8, 512]
        embedding = self.dropout(embedding_scaled + self.pos_embedding(source_pos))
        # print("pos_embedding: ", self.pos_embedding(source_pos).size())
        # [128, 8, 512]
        # print("embedding(pos + token): ", embedding.size())
        # print("B", embedding_scaled + self.pos_embedding(source_pos))
        # [127, 8, 512]
        # [batch_size, source_length, hidden_dim]

        for encoder_layer in self.encoder_layers:
            embedding = encoder_layer(embedding, source_mask)
        # [batch_size, source_length, hidden_dim]

        # print("Encoder output: ", embedding.size())
        # [128, 8, 512]

        return self.layer_norm(embedding) # todo 마지막에 layer norm을 더해주는듯?

#####################
# Decoder
#####################

class DecoderLayer(nn.Module):

    def __init__(self, args):
        super(DecoderLayer, self).__init__()

        self.args = args
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(args)
        self.encoder_attention = MultiHeadAttention(args)
        self.position_wise_ffn = PositionWiseFeedForward(args)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # print("----[Decoder_layer] * (default:6번)----")
        # print("target: ", target.size())
        # [128, 22 512]
        # print("encoder_output: ", encoder_output.size())
        # [128, 8, 512]
        # print("target_mask: ", target_mask.size())
        # [128, 22, 22]
        # print("dec_enc_mask: ", dec_enc_mask.size())
        # [128, 22, 8]
        # target [batch_size, target_length, hidden_dim]
        # encoder_output [batch_size, source_length, hidden_dim]
        # target_mask [batch_size, target_length, target_length]
        # dec_enc_mask [batch_size, target_length, source_length]

        # 여기서도 마찬가지로 LayerNorm(x + sublayer(x)) ==> x + SubLayer(LayerNorm(x))
        norm_target = self.layer_norm(target)
        # print("norm_target: ", norm_target.size())
        # [128, 22, 512]
        self_attention = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]
        # print("Decoder_self_attention_output: ", self_attention.size())
        # [128, 22, 512]

        # In Decoder part, Q: 이전의 layer 출력, K/V: last layer Encoder 출력
        normalized_self_attention = self.layer_norm(self_attention)
        # print("norm_self_attention: ", normalized_self_attention.size())
        # [128, 22, 512]
        sub_layer, attn_map = self.encoder_attention(normalized_self_attention,
                                                     encoder_output, encoder_output, dec_enc_mask)
        # print("dec_enc_attention_output: ", sub_layer.size())
        # return output, attentions score
        sub_layer_output = self_attention + sub_layer

        norm_sub_layer_norm = self.layer_norm(sub_layer_output)
        # print("layer_norm: ", norm_sub_layer_norm.size())
        output = sub_layer_output + self.position_wise_ffn(norm_sub_layer_norm)
        # output [batch_size, target_length, hidden_dim]

        return output, attn_map

class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        # hidden_dim 512 / output_dim 19545 [len(eng.vocab)]
        self.token_embedding = nn.Embedding(args.output_dim, args.hidden_dim, padding_idx=args.pad_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=args.hidden_dim ** -0.5)

        self.embedding_scale = args.hidden_dim ** 0.5
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_position_encoding(args.max_len + 1, args.hidden_dim, args), freeze=True
        )

        self.decoder_layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)

    def forward(self, target, source, encoder_output):
        # print("----[Decoder]----")
        # print("target: ", target.size())
        # [128, 22] <eos> 제외
        # print("source: ", source.size())
        # [128, 8]
        # print("encoder_output: ", encoder_output.size())
        # [128, 8, 512]

        # target [batch_size, target_length]
        # source [batch_size, source_length]
        # encoder_output [batch_size, source_length, hidden_dim]
        target_mask, dec_enc_mask = create_target_mask(source, target, self.args)
        # print("target_mask: ", target_mask.size())
        # [128, 22, 22]
        # print("dec_enc_mask: ", dec_enc_mask.size())
        # [128, 22, 8]
        # 둘 다 = [batch_size, target_length, target/source length]

        target_pos = create_position_vector(target, self.args)
        # [batch_size, target_length]

        embedding_scaled = self.token_embedding(target) * self.embedding_scale
        # print("token_embedding: ", embedding_scaled.size())
        # [128, 22, 512]
        embedding = self.dropout(embedding_scaled + self.pos_embedding(target_pos))
        # print("pos_embedding: ", self.pos_embedding(target_pos).size())
        # [128, 22, 512]
        # print("embedding(pos+token): ", embedding.size())
        # [128, 22, 512]
        # [batch_size, target_length, hidden_dim]

        for decoder_layer in self.decoder_layers:
            target, attn_map = decoder_layer(embedding, encoder_output, target_mask, dec_enc_mask)
        # target [batch_size, target_length, hidden_dim]
        # print("Decoder output: ", target.size())
        # [128, 22, 512]

        target_norm = self.layer_norm(target)
        # print("target_nrom: ", target_norm.size())
        # [128, 22, 512]

        output = torch.matmul(target_norm, self.token_embedding.weight.transpose(0, 1))
        # print("output: ", output.size())
        # [128, 22, 19545]
        # mm/bmm 의 차이점으로는 matmul: broadcast 지원
        return output, attn_map

class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, source, target):
        # source [batch_size, source_length] = [128, 6]
        # target [batch_size, target_length] = [128, 24]

        encoder_output = self.encoder(source)
        # [batch_size, source_length, hidden_dim]
        output, attn_map = self.decoder(target, source, encoder_output)
        # [batch_size, target_length, output_dim]

        return output, attn_map

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        # torch.numel(input): input tensor element 총 숫자
        # torch.randn(1, 2, 3, 4, 5) ==> 120














































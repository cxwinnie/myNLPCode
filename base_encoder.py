# coding: utf-8
import torch
import torch.nn as nn
import math
from word_tokenizer import WordTokenizer


class BaseEncoder(nn.Module):

    def __init__(self,
                 word2id,
                 word2vec,
                 max_length=128,
                 hidden_size=230,
                 position_size=5,
                 blank_padding=True,
                 mask_entity=False
                 ):
        super.__init__()
        self.word2id = word2id
        self.max_length = max_length
        self.word_num = len(word2id)
        self.mask_entity = mask_entity
        self.hidden_size = hidden_size
        self.position_size = position_size
        self.word_size = word2vec.shape[-1]
        self.input_size = self.word_size + position_size * 2
        self.blank_padding = blank_padding

        if not '[UNK]' in self.word2id:
            self.word2id['[UNK]'] = len(self.word2id)
            self.word_num += 1
        if not '[PAD]' in self.word2id:
            self.word2id['[PAD]'] = len(self.word2id)
            self.word_num += 1

        # word embedding
        self.word_embedding = nn.Embedding(self.word_num, self.word_size)
        word2vec = torch.from_numpy(word2vec)
        if self.word_num == len(word2vec) + 2:
            unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
            blk = torch.zeros(1, self.word_size)
            self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
        else:
            self.word_embedding.weight.data.copy_(word2vec)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.wordTokenizer = WordTokenizer(self.word2id)
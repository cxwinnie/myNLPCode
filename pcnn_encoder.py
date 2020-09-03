# coding: utf-8
from base_encoder import BaseEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch


class PCNNEncoder(BaseEncoder):
    def __init__(self,
                 word2id,
                 word2vec,
                 max_length=128,
                 hidden_size=230,
                 position_size=5,
                 blank_padding=True,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0.0,
                 activation_fcuntion=F.relu,
                 mask_entity=False):
        super().__init__(word2id, word2vec, max_length, hidden_size, position_size, blank_padding, mask_entity)

        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_fcuntion

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(
            torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100
        self.hidden_size *= 3  # 由于有hidden_size个filter，由三段最大化可知最后的长度为hidden_size*3

    def tokenize(self, bag):

        sentencts = bag['sents']
        sub = bag['sub']
        obj = bag['obj']
        seqs = None
        for sentence in sentencts:
            head = sentence.find(sub)
            tail = sentence.find(obj)
            head = len(sentence[:head].split())
            tail = len(sentence[:tail].split())
            pos_head = [head, head + len(sub.split())]
            pos_tail = [tail, tail + len(obj.split())]
            tokens = sentence.split()
            if pos_head[0] > pos_tail[0]:
                pos_head, pos_tail = pos_tail, pos_head
            if self.blank_padding:
                index_tokens = self.wordTokenizer.convert_tokens_to_ids(tokens, self.max_length,
                                                                        self.word2id['[PAD]'],
                                                                        self.word2id['[UNK]'])
            pos1 = []
            pos2 = []
            pos1_in_index = min(pos_head[0], self.max_length)
            pos2_in_index = min(pos_tail[0], self.max_length)
            for i in range(len(tokens)):
                pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
                pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

            if self.blank_padding:
                while len(pos1) < self.max_length:
                    pos1.append(0)
                while len(pos2) < self.max_length:
                    pos2.append(0)

            index_tokens = torch.tensor(index_tokens).long().unsqueeze(0)
            pos1 = torch.tensor(pos1).long().unsqueeze(0)
            pos2 = torch.tensor(pos2).long().unsqueeze(0)

            # mask
            mask = []
            pos_min = min(pos1_in_index, pos2_in_index)
            pos_max = max(pos1_in_index, pos2_in_index)
            for i in range(len(tokens)):
                if i <= pos_min:
                    mask.append(1)
                elif i <= pos_max:
                    mask.append(2)
                else:
                    mask.append(3)
            if self.blank_padding:
                while len(mask) < self.max_length:
                    mask.append(0)
            mask = torch.tensor(mask).long().unsqueeze(0)
            if seqs == None:
                seqs = []
                for i in range(4):
                    seqs.append([])
            seqs[0].append(index_tokens)
            seqs[1].append(pos1)
            seqs[2].append(pos2)
            seqs[3].append(mask)
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)
        return seqs

    def forward(self, token, pos1, pos2, mask):

        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)
        x = x.transpose(1, 2)
        x = self.conv(x)
        mask = self.mask_embedding(mask).transpose(1, 2)
        pool1 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))
        pool3 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))
        x = torch.cat([pool1, pool2, pool3], 1)
        x = x.squeeze(2)
        x = self.drop(x)

        return x

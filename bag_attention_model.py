# coding: utf-8
import torch
import torch.nn as nn
from base_model import BagRE


class BagAttention(BagRE):

    def __init__(self, sentence_encoder, num_rel, rel2id):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_rel = num_rel
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_rel)
        self.rel2id = rel2id
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(-1)
        self.id2rel = {v: k for k, v in rel2id.items()}

    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=0):

        token = token.view(-1, token.size(-1))
        pos1 = pos1.view(-1, pos1.size(-1))
        pos2 = pos2.view(-1, pos2.size(-1))

        mask = mask.view(-1, mask.size(-1))
        rep = self.sentence_encoder(token, pos1, pos2, mask)

        if train:
            batch_size = label.size(0)
            query = label.unsqueeze(1)
            att_mat = self.fc.weight.data[query]
            rep = rep.view(batch_size, bag_size, -1)
            att_score = (rep * att_mat).sum(-1)
            softmax_att_score = self.softmax(att_score)
            bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep)
        else:
            batch_size = rep.size(0) // bag_size
            att_score = torch.matmul(rep, self.fc.weight.data)
            att_score = att_score.view(batch_size,bag_size,-1)
            rep = rep.view(batch_size, bag_size, -1)
            softmax_att_score = self.softmax(att_score.transpose(1,2))
            rep_for_each_rel = torch.matmul(softmax_att_score,rep)
            bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2)

        return bag_logits
# coding: utf-8
import torch
import torch.nn as nn
from base_model import BagRE


class BagAttention(BagRE):

    def __init__(self, sentence_encoder, num_rel, rel2id, dropout=0):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_rel = num_rel
        self.att_w = nn.ParameterList(
            [nn.Parameter(torch.eye(self.sentence_encoder.hidden_size)) for _ in range(num_rel)])
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_rel)
        self.rel_bias = nn.Parameter(torch.rand(self.num_rel))
        self.rel2id = rel2id
        self.drop = nn.Dropout(dropout)
        self.softmaxAtt = nn.Softmax(0)
        self.softLogist = nn.Softmax(1)
        self.id2rel = {v: k for k, v in rel2id.items()}

    def forward(self, labels, scope, token, pos1, pos2, mask=None,train = True):

        # 生成每个句子的最终embedding（由token，POS1，POS2和MASK生成）,如果一个包中有多个句子，那么这个包就会生成多个句子对应的embedding
        bag_features = self.sentence_encoder(scope, token, pos1, pos2, mask)

        bag_reps = []
        if train:
            for bag_embs, label in zip(bag_features, labels):
                att_mat = self.att_w[label]
                att_score = bag_embs.mm(att_mat).mm(self.fc.weight.data[label].view(-1, 1))
                bag_embs = bag_embs * self.softmaxAtt(att_score)
                bag_rep = torch.sum(bag_embs, 0)
                bag_rep = self.drop(bag_rep)
                bag_reps.append(bag_rep.unsqueeze(0))
            bag_reps = torch.cat(bag_reps, 0)
            bag_logits = self.fc(bag_reps) + self.rel_bias
            return bag_logits
        else:
            pre_y = []
            for label in range(0, self.num_rel):
                bag_reps = []
                labels = [label for _ in range(len(bag_features))]
                for bag_embs, label in zip(bag_features, labels):
                    att_mat = self.att_w[label]
                    att_score = bag_embs.mm(att_mat).mm(self.fc.weight.data[label].view(-1, 1))
                    bag_embs = bag_embs * self.softmaxAtt(att_score)
                    bag_rep = torch.sum(bag_embs, 0)
                    bag_rep = self.drop(bag_rep)
                    bag_reps.append(bag_rep.unsqueeze(0))
                bag_reps = torch.cat(bag_reps, 0)
                bag_logits = self.fc(bag_reps) + self.rel_bias
                pre_y.append(bag_logits.unsqueeze(1))
            res = torch.cat(pre_y, 1).max(1)[0]
            return self.softLogist(res)
# coding: utf-8
import torch
import torch.utils.data as data
import numpy as np
import random
import sklearn
from sklearn import metrics


class BagREDataset(data.Dataset):

    def __init__(self, path, rel2id, tokenize, entpair_as_bag=False, mode=None):
        super().__init__()
        self.tokenize = tokenize
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag

        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
        for idx, item in enumerate(self.data):
            fact = (item['subId'], item['objId'], item['rel'])
            if item['rel'] == 0:
                self.facts[fact] = 1
            if entpair_as_bag:
                name = (item['subId'], item['objId'])
            else:
                name = fact
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.bag_scope.append([])
                self.bag_name.append(name)
            self.bag_scope[self.name2id[name]].append(idx)
            self.weight[item['rel']] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag_index = self.bag_scope[index]
        bag = self.data[bag_index[0]]
        rel = bag['rel']
        seqs = list(self.tokenize(bag))
        return [rel, self.bag_name[index], len(bag['sents'])] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            tmp = []
            for emb in seqs[i]:
                tmp.append(emb)
            seqs[i] = tmp
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()
        return [label, bag_name, scope] + seqs

    def eval(self, true_y, pred_y, pred_p):
        pos_num = len([i for i in true_y if i > 0])
        index = np.argsort(pred_p)[::-1]

        tp = 0
        fn = 0
        fp = 0

        all_pre = [0]
        all_rec = [0]

        for idx in range(len(index)):
            i = true_y[index[idx]]
            j = pred_y[index[idx]]
            if i == 0:
                if j > 0:
                    fp += 1
            else:
                if j == 0:
                    fn += 1
                else:
                    if i == j:
                        tp += 1

            if fp + tp == 0:
                precision = 1
            else:
                precision = float(tp) / (fp + tp)

            recall = float(tp) / (pos_num+1e-20)

            if precision != all_pre[-1] or recall != all_rec[-1]:
                all_pre.append(precision)
                all_rec.append(recall)

        auc = metrics.auc(x=all_rec, y=all_pre)

        np_prec = np.array(all_pre[1:])
        np_rec = np.array(all_rec[1:])

        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()

        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}

def BagRELoader(path, rel2id, tokenize, batch_size,
                shuffle, entpair_as_bag=False, bag_size=0, num_workers=0):
    collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenize, entpair_as_bag=entpair_as_bag)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader

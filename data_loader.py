# coding: utf-8
import torch
import torch.utils.data as data
import numpy as np
import random
import sklearn

class BagREDataset(data.Dataset):

    def __init__(self, path, rel2id, tokenize, entpair_as_bag=False, bag_size=0, mode=None):
        super().__init__()
        self.tokenize = tokenize
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

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
        if self.bag_size > 0:
            if self.bag_size <= len(bag['sents']):
                resize_bag = random.sample(bag['sents'], self.bag_size)
            else:
                resize_bag = bag['sents'] + list(np.random.choice(bag['sents'], self.bag_size - len(bag['sents'])))
            bag['sents'] = resize_bag

        rel = bag['rel']
        seqs = list(self.tokenize(bag))
        return [rel, self.bag_name[index], len(bag['sents'])] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result):
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i,item in enumerate(sorted_pred_result):
            if(item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct)/float(i+1))
            rec.append(float(correct)/float(total))
        auc = sklearn.metrics.auc(x=rec,y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_rec * np_prec/(np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def BagRELoader(path, rel2id, tokenize, batch_size,
                shuffle, entpair_as_bag=False, bag_size=0, num_workers=0):
    collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path,rel2id,tokenize,entpair_as_bag=entpair_as_bag,bag_size=bag_size)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader

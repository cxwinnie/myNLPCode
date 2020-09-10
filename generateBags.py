# coding: utf-8
import json
from collections import defaultdict
import os
import numpy as np

'''rel2id = json.loads(open('./dataset/riedel_relation2id.json').read())
id2rel = dict([(v, k) for k, v in rel2id.items()])


处理riedel数据集

print('Constructing training bags...')
train_data = defaultdict(lambda: {'rels': defaultdict(list)})
test_data = defaultdict(lambda: {'rels': defaultdict(list)})

print('Constructing train bags...')
with open('dataset/riedel_train.json') as f:
    for _, line in enumerate(f):
        data = json.loads(line.strip())
        id = '{}_{}'.format(data['sub'], data['obj'])  # 头实体和尾实体组成的id
        train_data[id]['sub'] = data['sub']
        train_data[id]['subId'] = data['sub_id']
        train_data[id]['obj'] = data['obj']
        train_data[id]['objId'] = data['obj_id']
        train_data[id]['rels'][rel2id.get(data['rel'], rel2id.get('NA'))].append(data['rsent'])

print('Constructing test bags...')
with open('dataset/riedel_test.json') as f:
    for _, line in enumerate(f):
        data = json.loads(line.strip())
        id = '{}_{}'.format(data['sub'], data['obj'])  # 头实体和尾实体组成的id
        test_data[id]['sub'] = data['sub']
        test_data[id]['subId'] = data['sub_id']
        test_data[id]['obj'] = data['obj']
        test_data[id]['objId'] = data['obj_id']
        test_data[id]['rels'][rel2id.get(data['rel'], rel2id.get('NA'))].append(data['rsent'])

print("create train_bags.json")
with open('dataset/riedel_train_bag.json', 'w') as f:
    for id, data in train_data.items():
        for rel, sents in data['rels'].items():
            entry = {}
            entry['sub'] = data['sub']
            entry['subId'] = data['subId']
            entry['obj'] = data['obj']
            entry['objId'] = data['objId']
            entry['rel'] = rel
            entry['sents'] = sents
            f.write(json.dumps(entry) + '\n')

print("create test_bags.json")
with open('dataset/riedel_test_bag.json', 'w') as f:
    for id, data in test_data.items():
        for rel, sents in data['rels'].items():
            entry = {}
            entry['sub'] = data['sub']
            entry['subId'] = data['subId']
            entry['obj'] = data['obj']
            entry['objId'] = data['objId']
            entry['rel'] = rel
            entry['sents'] = sents
            f.write(json.dumps(entry) + '\n')

word_list = []
vecs = []
with open('dataset/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        vec = list(map(float, line[1:]))
        vecs.append(vec)
        word_list.append(line[0])
word2id = {j: i for i, j in enumerate(word_list)}
id2word = {int(i): j for i, j in enumerate(word_list)}
np.save('dataset/w2v.npy', vecs)
with open('dataset/id2word.json', 'w') as f:
    f.write(json.dumps(id2word))
'''

'''
处理OpenNRE数据集
'''
train_data = defaultdict(lambda: {'rels': defaultdict(list)})
test_data = defaultdict(lambda: {'rels': defaultdict(list)})
rel2id = json.loads(open('./nyt10/nyt10_rel2id.json').read())
id2rel = dict([(v, k) for k, v in rel2id.items()])
print('Constructing train bags...')
with open('nyt10/nyt10_train.txt') as f:
    for _, line in enumerate(f):
        data = json.loads(line.strip())
        id = '{}_{}'.format(data['h']['id'], data['h']['id'])  # 头实体和尾实体组成的id
        train_data[id]['sub'] = data['h']['name']
        train_data[id]['subId'] = data['h']['id']
        train_data[id]['obj'] = data['t']['name']
        train_data[id]['objId'] = data['t']['id']
        train_data[id]['rels'][rel2id.get(data['relation'], rel2id.get('NA'))].append(data['text'])

print("create train_bags.json")
with open('nyt10/nyt10_train_bag.json', 'w') as f:
    for id, data in train_data.items():
        for rel, sents in data['rels'].items():
            entry = {}
            entry['sub'] = data['sub']
            entry['subId'] = data['subId']
            entry['obj'] = data['obj']
            entry['objId'] = data['objId']
            entry['objId'] = data['objId']
            entry['rel'] = rel
            entry['sents'] = sents
            f.write(json.dumps(entry) + '\n')

print('Constructing test bags...')
with open('nyt10/nyt10_test.txt') as f:
    for _, line in enumerate(f):
        data = json.loads(line.strip())
        id = '{}_{}'.format(data['h']['id'], data['h']['id'])  # 头实体和尾实体组成的id
        test_data[id]['sub'] = data['h']['name']
        test_data[id]['subId'] = data['h']['id']
        test_data[id]['obj'] = data['t']['name']
        test_data[id]['objId'] = data['t']['id']
        test_data[id]['rels'][rel2id.get(data['relation'], rel2id.get('NA'))].append(data['text'])

print("create test_bags.json")
with open('nyt10/nyt10_test_bag.json', 'w') as f:
    for id, data in test_data.items():
        for rel, sents in data['rels'].items():
            entry = {}
            entry['sub'] = data['sub']
            entry['subId'] = data['subId']
            entry['obj'] = data['obj']
            entry['objId'] = data['objId']
            entry['rel'] = rel
            entry['sents'] = sents
            f.write(json.dumps(entry) + '\n')
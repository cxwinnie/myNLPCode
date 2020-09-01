# coding: utf-8
import os
import numpy as np
import torch
from config import opt
import json
from pcnn_encoder import PCNNEncoder
from bag_attention_model import BagAttention
from bag_re import BagRE

torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

rel2id = json.load(open('dataset/riedel_relation2id.json'))
id2word = json.load(open('dataset/id2word.json'))
word2id = {j: int(i) for i, j in id2word.items()}
word2v = np.load(open('dataset/w2v.npy'))
ckpt = 'ckpt/nyt10_pcnn_att.pth'

if opt.use_gpu:
    torch.cuda.set_device(opt.gpu_id)

sentence_encoder = PCNNEncoder(
    word2id=word2id,
    max_length=120,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2v,
    dropout=0.5
)

model = BagAttention(sentence_encoder, len(rel2id), rel2id)

framework = BagRE(
    train_path='dataset/riedel_train_bag.json',
    test_path='dataset/riedel_test_bag.json',
    model = model,
    ckpt=ckpt,
    batch_size=160,
    max_epoch=60,
    lr=0.5,
    weight_decay=0,
    opt='sgd',
    bag_size=10
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('AUC on test set: {}'.format(result['auc']))

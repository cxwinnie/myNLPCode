# coding: utf-8 

from torch import nn, optim
from data_loader import *
from tqdm import tqdm
from utils import AverageMeter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')


class BagRE(nn.Module):
    def __init__(self,
                 model,
                 train_path,
                 test_path,
                 ckpt,
                 batch_size=32,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False):
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        if train_path is not None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                bag_size=bag_size,
                entpair_as_bag=False
            )

        if test_path is not None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=False
            )

        self.model = nn.DataParallel(model)

        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            self.optimizer = optim.AdamW(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")

        if torch.cuda.is_available():
            self.cuda()

        self.ckpt = ckpt

    def train_model(self):
        best_auc = 0
        for epoch in range(self.max_epoch):
            # train
            self.train()
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            print("=== Epoch %d train ===" % epoch)
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[1]
                args = data[3:]
                logits = self.model(label, scope, *args, )
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0)).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grag()

            # Val
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("auc: %.4f" % result['auc'])
            print("f1: %.4f" % (result['f1']))
            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_auc = result['auc']
        print("Best auc on val set: %f" % (best_auc))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False)
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_rel):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.module.id2rel[relid],
                                'score': logits[i][relid]
                            })
            result = eval_loader.dataset.eval(pred_result)

            prec, rec = result['prec'], result['rec']
            plt.ylim([0.3, 1.0])  # y的范围从0.3到1
            plt.xlim([0.0, 0.45])  # x的范围从0到0.45
            plt.plot(rec, prec, label='MyTest', color='red', lw=1, marker='o', markevery=0.1, ms=6)
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.legend(loc="upper right", prop={'size': 12})  # 将图例信息设置在右上角
            plt.grid(True)
            plt.tight_layout()  # 当你拥有多个子图时，你会经常看到不同轴域的标签叠在一起，tight_layout()也会调整子图之间的间隔来减少堆叠
            plt.show()

            plot_path = './plot_pr.pdf'
            plt.savefig(plot_path)

            return result

import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import collections
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    top_k_accuracy_score, \
    mean_squared_error, r2_score
from sklearn.model_selection import KFold
from transformers import AutoModel
from torch_geometric.nn import GATConv, GCNConv
from collections import Counter
from torch_geometric.data import Data


class SVFeatureBlock(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, mode='mean'):
        super(SVFeatureBlock, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size

        if mode == 'lstm':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
            nn.init.orthogonal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
        elif mode == 'bi-lstm':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True,
                                bidirectional=True)
        elif self.mode == "gru":
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif mode == 'rnn':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, sv):
        sv_list = []
        for x_tmp in sv:  # 拆解 batch 这个维度
            if self.mode == "mean":
                if x_tmp.dim() != 1:
                    out_put = torch.mean(x_tmp, dim=0)  # 按列求均值，得到(512)
            elif self.mode == "sum":
                if x_tmp.dim() != 1:
                    out_put = torch.sum(x_tmp, dim=0)  # 按列求和
            elif self.mode == "max":
                if x_tmp.dim() != 1:
                    out_put = torch.max(x_tmp, dim=0).values  # 按列求max
            elif self.mode == "lstm":
                out_put, (h_n, c_n) = self.lstm(x_tmp.view(1, -1, self.input_size))  # 输入 lstm 需要加上 batch 这个维度
                out_put = out_put[:, -1, :]  # 只取最后一层的输出 # （1,1,512)
                out_put = torch.squeeze(out_put)  # 得到 (512)
            elif self.mode == "gru":
                out_put, h_n = self.gru(x_tmp.view(1, -1, self.input_size))  # 输入 lstm 需要加上 batch 这个维度
                out_put = out_put[:, -1, :]  # 只取最后一层的输出 # （1,1,512)
                out_put = torch.squeeze(out_put)  # 得到 (512)
            elif self.mode == "rnn":
                out_put, h_n = self.rnn(x_tmp.view(1, -1, self.input_size))  # 输入 lstm 需要加上 batch 这个维度
                out_put = out_put[:, -1, :]  # 只取最后一层的输出 # （1,1,512)
                out_put = torch.squeeze(out_put)  # 得到 (512)
            else:
                pass

            sv_list.append(out_put)
        x = torch.stack(sv_list)  # 拼接,(batch,512)
        return x


class SoftAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftAttention, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim, bias=False)
        self.layer2 = nn.Linear(input_dim, output_dim, bias=True)
        self.layer3 = nn.Sigmoid()
        self.c = nn.Parameter(torch.randn(output_dim))

    def forward(self, sv):  # (10,12,64)-->(10,64)
        outs = []
        for street in sv:
            c_t = self.c.unsqueeze(1)  # Transpose c
            st_mean = torch.mean(street, dim=0)
            a = self.layer1(st_mean)
            b = self.layer2(street)
            a_b = self.layer3(a + b)
            alpha = torch.matmul(a_b, c_t).squeeze(1)

            output = torch.matmul(alpha, street)
            outs.append(output)

        outs = torch.stack(outs, 0)
        return outs


def weights_init_1(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)


def weights_init_2(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)


class Attention_Soft(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention_Soft, self).__init__()

        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Sigmoid()
        self.l2 = torch.nn.Linear(in_size, hidden_size, bias=False)
        self.l3 = torch.nn.Linear(int(hidden_size), 1, bias=False)
        # self.l4 = torch.nn.Linear(in_size, 1024, bias=True)

        weights_init_2(self.l1)
        weights_init_1(self.l2)
        weights_init_1(self.l3)

    def forward(self, z):
        w1 = self.l1(torch.mean(z, dim=1).unsqueeze(1))
        w2 = self.l2(z)
        w = self.ac(w1 + w2)
        w = self.l3(w)
        beta = torch.softmax(w, dim=1)

        # z=self.l4(z)
        return (beta * z).sum(1)


class SV_GAT(nn.Module):
    def __init__(self, args):
        super(SV_GAT, self).__init__()
        self.args = args
        self.task = args.task
        print(self.task)

        self.length = list(np.load('data/length.npy'))
        # pretrained embedding
        pretrain_sv_path = args.pretrain_sv_path
        pretrain_scn_path = args.pretrain_scn_path
        print(pretrain_sv_path)
        print(pretrain_scn_path)
        # self.sv_embedding = torch.load(pretrain_sv_path, map_location=torch.device(args.device))
        self.sv_embedding = torch.load(pretrain_sv_path, map_location=torch.device(args.device)).to(torch.float32)
        self.scn_embedding = torch.load(pretrain_scn_path, map_location=torch.device(args.device))
        # self.sv_embedding_shuffled = self.shuffled()
        # print(f'shuffle={self.sv_embedding_shuffled.shape}')

        # Resnet
        model_simCLR_resnet18_path = "model/checkpoint_60_pretrain_h_0125.tar"
        print(model_simCLR_resnet18_path)
        mlp_pretrain_sv = torch.load(model_simCLR_resnet18_path, map_location=torch.device(args.device))
        self.projector_sv = nn.Sequential(
            nn.Linear(args.dim_size, args.dim_size),
            nn.ReLU(),
        )
        self.fc_sv = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.sv_agg = SVFeatureBlock(input_size=768, hidden_size=768, mode=args.mode)

        print(args.mode)

        # BERT
        self.projector_scn = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
        )

        self.attention_soft = Attention_Soft(in_size=768, )

        self.gat = GAT(input_dim=768, hidden_dim=64, output_dim=10, heads=8, args=args, drop=args.drop)

        self.gat_poi = GAT_P(input_dim=768, hidden_dim=64, output_dim=4, heads=8, args=args)

    def forward(self):
        sv_features = self.sv_embedding
        street_list = list(torch.split(sv_features, self.length, dim=0))

        sv_aggre = self.sv_agg(street_list)
        sv_embedding = sv_aggre
        scn_embedding = self.scn_embedding  # 768->768

        street_embedding = self.attention_soft(torch.stack([scn_embedding, sv_embedding], dim=1))

        if self.args.downstream == 'poi':
            gat_loss, out = self.gat_poi(street_embedding)  # Perform a single forward pass.
        else:
            gat_loss, s_emb1,out = self.gat(street_embedding)  # Perform a single forward pass.

        loss = gat_loss

        return loss, out, street_embedding

    def test(self, out):
        if self.args.downstream == 'poi':
            acc, f1_score_test, mrr_test, num, pred_out = self.gat_poi.test(out)
            return acc, f1_score_test, mrr_test, 1, 1, 1, num, pred_out
        else:
            a1, a3, a5, a10, f1, mrr, num, pred_out = self.gat.test(out)
            return a1, a3, a5, a10, f1, mrr, num, pred_out


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads, args, drop=0.6):
        super().__init__()
        torch.manual_seed(args.seed2)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, concat=False,
                             heads=10, dropout=0.6)
        self.elu = nn.ELU()
        self.drop1 = nn.Dropout(p=drop, )
        self.drop2 = nn.Dropout(p=0.6, )

        self.edge_index = torch.load('data/edge_index.pt').t().contiguous().to(args.device)
        self.y = torch.from_numpy(np.load('data/label_all_function.npy', allow_pickle=True)).long().to(args.device)
        self.train_mask = torch.from_numpy(np.load(args.mask_f, allow_pickle=True)).to(args.device)
        self.test_mask = torch.from_numpy(np.load(args.mask_f, allow_pickle=True)).to(args.device)

        self.mask = torch.load(args.mask_f_test)

        self.y_testlabel = np.load('data/label_all_function.npy')[self.mask]

    def forward(self, street_embedding):
        street_embedding = self.drop1(street_embedding)
        street_embedding_1 = self.conv1(street_embedding, self.edge_index)
        street_embedding_2 = self.elu(street_embedding_1)
        street_embedding_2 = self.drop2(street_embedding_2)
        street_embedding_2 = self.conv2(street_embedding_2, self.edge_index)

        cross_criterion = torch.nn.CrossEntropyLoss()
        loss_su = cross_criterion(street_embedding_2[self.train_mask], self.y[self.train_mask])

        return loss_su, street_embedding_1, street_embedding_2

    def test(self, out):
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred = pd.DataFrame({'Type': torch.Tensor.cpu(pred).numpy()})

        predictions_test_dim = torch.Tensor.cpu(out[self.mask]).argmax(dim=1).detach().numpy()
        predictions_test = torch.Tensor.cpu(out[self.mask]).detach().numpy()

        A1 = top_k_accuracy_score(self.y_testlabel, predictions_test, k=1, labels=range(10))
        A3 = top_k_accuracy_score(self.y_testlabel, predictions_test, k=3, labels=range(10))
        A5 = top_k_accuracy_score(self.y_testlabel, predictions_test, k=5, labels=range(10))
        print(f'A1={A1}\t A3={A3}\t A5={A5} ')

        precision_score_test = precision_score(self.y_testlabel, predictions_test_dim, average="weighted")
        f1_score_test = f1_score(self.y_testlabel, predictions_test_dim, average="weighted")
        mrr_test = compute_mrr(self.y_testlabel, predictions_test)
        result = Counter(pred['Type'].values.tolist())
        num = len(result)
        print(
            f'precision={precision_score_test}, f1={f1_score_test}, mrr={mrr_test},num={num}')

        print(result)
        return A1, A3, A5, 1, f1_score_test, mrr_test, num, out


class GAT_P(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads, args, drop=0.6):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=drop)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, concat=False,
                             heads=4, dropout=drop)
        self.elu = nn.ELU()
        self.drop1 = nn.Dropout(p=drop, )
        self.drop2 = nn.Dropout(p=drop, )

        self.edge_index = torch.load('data/edge_index.pt').t().contiguous().to(args.device)

        self.y = torch.from_numpy(np.load('data/label_all_poi_level.npy', allow_pickle=True)).long().to(args.device)
        self.train_mask = torch.from_numpy(np.load(args.mask_p, allow_pickle=True)).to(args.device)
        self.test_mask = torch.from_numpy(np.load(args.mask_p_test, allow_pickle=True)).to(args.device)

        self.mask = torch.from_numpy(np.load(args.mask_p_test, allow_pickle=True))

        self.y_testlabel = np.load('data/label_all_poi_level.npy')[self.mask]

    def forward(self, street_embedding):
        street_embedding = self.drop1(street_embedding)
        street_embedding = self.conv1(street_embedding, self.edge_index)
        street_embedding = self.elu(street_embedding)
        street_embedding = self.drop2(street_embedding)
        street_embedding = self.conv2(street_embedding, self.edge_index)

        cross_criterion = torch.nn.CrossEntropyLoss()
        loss_su = cross_criterion(street_embedding[self.train_mask], self.y[self.train_mask])

        return loss_su, street_embedding

    def test(self, out):
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[self.test_mask] == self.y[self.test_mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(self.test_mask.sum())  # Derive ratio of correct predictions.

        pred = pd.DataFrame({'Type': torch.Tensor.cpu(pred).numpy()})

        predictions_test_dim = torch.Tensor.cpu(out[self.mask]).argmax(dim=1).detach().numpy()
        predictions_test = torch.Tensor.cpu(out[self.mask]).detach().numpy()
        f1_score_test = f1_score(self.y_testlabel, predictions_test_dim, average="macro")
        mrr_test = compute_mrr(self.y_testlabel, predictions_test)
        result = Counter(pred['Type'].values.tolist())
        num = len(result)
        print(
            f'acc={acc}, f1={f1_score_test}, mrr={mrr_test},num={num}')

        print(result)
        return acc, f1_score_test, mrr_test, num, out






def compute_mrr(true_labels, machine_preds):
    """Compute the MRR """
    # print(true_labels[0])
    # print(machine_preds[0])
    rr_total = 0.0
    for i in range(len(true_labels)):
        if true_labels[i] == 403:
            continue
        ranklist = list(np.argsort(machine_preds[i])[::-1])  # 概率从大到小排序，返回index值
        rank = ranklist.index(true_labels[i]) + 1  # 获取真实值的rank
        rr_total = rr_total + 1.0 / rank
    mrr = rr_total / len(true_labels)
    return mrr


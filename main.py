import itertools
import os
import random

import math
from matplotlib.colors import ListedColormap
import pandas as pd
import torch
import numpy as np
import argparse
import warnings

from model import SV_GAT

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser.add_argument('--device', type=str, default='cuda:7', help='gpu device ids')
parser.add_argument('--print_num', type=int, default=10, help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')
parser.add_argument("--print_epoch", type=int, default=10, nargs="?", help="Start print epoch")
parser.add_argument("--start_epoch", type=int, default=0, nargs="?", help="Start epoch")
parser.add_argument("--current_epoch", type=int, default=0, nargs="?", help="Current epoch")
parser.add_argument("--global_step", type=int, default=0, nargs="?", help="global_step")
parser.add_argument("--epochs", type=int, default=1000, nargs="?", help="Epochs")
parser.add_argument("--seed", type=int, default=42, nargs="?", help="random seed.")

parser.add_argument("--mode", type=str, default='lstm', nargs="?", help="random seed.")
parser.add_argument("--task", type=str, default='attention_soft_best', nargs="?", help="random seed.")
parser.add_argument("--lstm_lr", type=float, default=0.005, nargs="?", help="Learning rate.")
parser.add_argument("--downstream", type=str, default='function', nargs="?", help="poi,function")
parser.add_argument("--pretrain_sv_path", type=str, default='embeddings/sv_huang117144_0128_test2_100.pt', nargs="?")
parser.add_argument("--pretrain_scn_path", type=str, default='embeddings/text_embedding_llama.pt', nargs="?",)
parser.add_argument("--dim_size", type=int, default=768, nargs="?",)
parser.add_argument("--drop", type=float, default=0.6, nargs="?",)
parser.add_argument("--decay", type=float, default=5e-4, nargs="?",)
parser.add_argument("--mask_f", type=str, default='data/label_mask.npy', nargs="?",)
parser.add_argument("--mask_f_test", type=str, default='test_mask.pt', nargs="?",)
parser.add_argument("--mask_p", type=str, default='data/label_mask_poi_level.npy', nargs="?",)
parser.add_argument("--mask_p_test", type=str, default='data/test_mask_poi_level.npy', nargs="?",)


args = parser.parse_args()

# args = parser.parse_known_args()[0]  # jupyter

class EarlyStopping:
    def __init__(self, patience=48, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_pred = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss, pred):
        score = loss
        if self.best_score is None:
            self.best_score = score
            self.best_pred = pred
            # self.save_checkpoint(model, path)
        elif score > self.best_score - self.delta * self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score <= self.best_score:
            self.best_score = score
            self.best_pred = pred
            # self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter = 0


def trainer(args, model, optimizer1, optimizer2, optimizer3, epoch):
    loss_epoch = []
    model.train()
    if True:
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            if args.model_name == 'Frozen_Encoder_NoContrastive':
                gnn_loss, pre_out, street_embedding = model()
                loss_epoch.append(gnn_loss.item())
                loss = gnn_loss

                loss.backward()

                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

    if epoch % 10 == 0:
        print(
            # f"TrainEpoch [{epoch + 1}/{args.epochs}]\t loss_epoch_con:{np.mean(loss_epoch_con)}\t loss_epoch_gnn:{np.mean(loss_epoch_gnn)}")
            f"TrainEpoch [{epoch + 1}/{args.epochs}]\t loss_epoch_gnn:{np.mean(loss_epoch)}")
    return np.mean(loss_epoch), pre_out, street_embedding





def test(args, model, epoch):
    with torch.no_grad():
        model.eval()
        # model2.eval()
        # global t_street_embedding
        # for step, batch in enumerate(train_loader):
        if True:
            with torch.autograd.set_detect_anomaly(True):
                if args.model_name == 'Frozen_Encoder_NoContrastive':
                    _, out ,_ = model()
                    if args.downstream == 'poi':
                        acc, f1, mrr, _,_,_,num, pred_out = model.test(out)
                        return acc, f1, mrr,1,1,1, num, pred_out
                    else:
                        a1, a3, a5, a10, f1, mrr, num, pred_out = model.test(out)
                        return a1, a3, a5, a10, f1, mrr, num, pred_out


def tsne(street_emb,epoch):
    # if epoch in [10,200,300,400,500,600,700,800,900]:
    if True:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np

        street_emb = torch.Tensor.cpu(street_emb).detach().numpy()
        tsne = TSNE(n_components=2, random_state=42,perplexity=10.0)
        embeddings_reduced = tsne.fit_transform(street_emb)

        colors = ['#FF0000', '#00FF00', '#0000FF', '#00C2FF', '#008000', '#FFA500', '#BA00FF', '#A1A1A1', '#F5FF00',
                  '#000000']
        my_cmap = ListedColormap(colors, name='my_cmap')
        label = np.load('data/label_all_function.npy')
        # x_min, x_max = np.min(data, 0), np.max(data, 0)
        # data = (data - x_min) / (x_max - x_min)
        # plt.figure()
        for i in range(embeddings_reduced.shape[0]):
            # pred_label = street_emb[i].argmax()
            plt.plot(embeddings_reduced[i, 0], embeddings_reduced[i, 1], marker='o', markersize=5, color=colors[label[i]])
        plt.xticks([])
        plt.yticks([])
        # plt.legend(range(7))
        plt.savefig('img/{}_my.pdf'.format(epoch), bbox_inches='tight', dpi=108, pad_inches=0.2)
        plt.show()


for i in range(8):

    np.random.seed(args.seed)
    random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed(args.seed + 3)
    torch.backends.cudnn.deterministic = True

    model = SV_GAT(args)

    model = model.to(args.device)

    # model2 = model2.to(args.device)

    # opt1 = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    opt1 = torch.optim.Adam(
        itertools.chain(model.projector_sv.parameters(), model.fc_sv.parameters(), model.cross_layer.parameters(), model.attention.parameters(),model.attention_soft.parameters(),
                        model.attn_layer.parameters(), model.projector_scn.parameters(), model.set2set.parameters(),
                        ), lr=0.0005, weight_decay=1e-8)
    if args.downstream == 'poi':
        opt3 = torch.optim.Adam(model.gat_poi.parameters(), lr=0.0005, weight_decay=5e-4)
    else:
        opt3 = torch.optim.Adam(model.gat.parameters(), lr=0.005, weight_decay=5e-4)
    print('opt1={}'.format(opt1.param_groups[0]['lr']))
    print('opt3={}'.format(opt3.param_groups[0]['lr']))
    if args.mode != 'mean':
        opt2 = torch.optim.SGD(model.sv_agg.parameters(), lr=args.lstm_lr, weight_decay=1e-4, momentum=0.9)
        print('opt2={}'.format(opt2.param_groups[0]['lr']))
        # opt3 = torch.optim.Adam(model.gcn.parameters(), lr=0.005, weight_decay=5e-4)
        # 学习率衰减
        t = 10  # warmup
        T = 800  # epochs - 10 为 cosine rate
        # lr_warm = 0.0035
        n_t = 0.5
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) /2 ) * (1 - args.lr_factor) + args.lr_factor # cosine
        lf = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lf)
    # opt2 = torch.optim.Adam(
    #     itertools.chain(
    #         model.parameters(), model2.parameters()
    #     ), lr=0.005, weight_decay=5e-4)

    # %%
    if args.task=='attention_soft_best_lstm':
        earlystop = EarlyStopping(delta=0.01,patience=48)
    else:
        earlystop = EarlyStopping(delta=0.01)
    print(model)
    print(args)


    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch, pred_ ,street_embedding = trainer(args, model, opt1, opt2, opt3, epoch)
        # if epoch in range(0, args.epochs, 20):
        # out = os.path.join(args.model_path, "checkpoint_train_1218_{}.tar".format(args.current_epoch))
        # torch.save(model.state_dict(), out)
        scheduler.step()
        if epoch in range(args.print_epoch, args.epochs + 10, 5):
            tsne(pred_, epoch)
            _, _, _, _, _, _, num, pred_out = test(args, model, epoch)
            if args.downstream=='function':
                if num == 10:
                    earlystop(loss=loss_epoch, pred=pred_)
            else:
                if num == 4:
                    earlystop(loss=loss_epoch, pred=pred_)
            print('lstm的lr={}'.format(opt2.param_groups[0]['lr']))


        args.current_epoch += 1
        # if epoch > 10:
        #     earlystop(loss=loss_epoch, pred=pred_)
        # if epoch==460:
        #     earlystop.early_stop = True
        if earlystop.early_stop:
            # print(earlystop.best_score)
            a1,a3,a5,a10, f1, mrr,num, _ = test(args, model, epoch)

            df = pd.read_csv('results/results2_.csv')
            # df.loc[len(df)]=[args.pretrain_sv_path[11:],args.pretrain_scn_path[11:],args.task,a1,a3,a5,a10,f1,mrr,num]
            df.loc[len(df)]=[args.pretrain_sv_path[11:],args.task,a1,a3,a5,a10,f1,mrr,num]
            df.to_csv('results/results2_.csv', index=False)

            a1,a3,a5,a10, f1, mrr,num, _=model.test(earlystop.best_pred)

            df = pd.read_csv('results/results2_.csv')
            # df.loc[len(df)] = [args.pretrain_sv_path[11:],args.pretrain_scn_path[11:],args.task, a1, a3, a5, a10, f1, mrr, num]
            df.loc[len(df)] = [args.pretrain_sv_path[11:],args.task, a1, a3, a5, a10, f1, mrr, num]
            df.to_csv('results/results2_.csv', index=False)

            print(args.current_epoch)
            break





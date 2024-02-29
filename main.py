import itertools
import os
import random

import math
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
parser.add_argument("--print_epoch", type=int, default=50, nargs="?", help="Start print epoch")
parser.add_argument("--start_epoch", type=int, default=0, nargs="?", help="Start epoch")
parser.add_argument("--current_epoch", type=int, default=0, nargs="?", help="Current epoch")
parser.add_argument("--epochs", type=int, default=150, nargs="?", help="Epochs")
parser.add_argument("--seed", type=int, default=42, nargs="?", help="random seed.")

parser.add_argument("--mode", type=str, default='lstm', nargs="?", help="aggression function.")
parser.add_argument("--pretrain_sv_path", type=str, default='embeddings/visual_embedding.pt', nargs="?")
parser.add_argument("--pretrain_scn_path", type=str, default='embeddings/text_embedding.pt', nargs="?")
parser.add_argument("--downstream", type=str, default='function', nargs="?", help="poi,function")

args = parser.parse_args()


def trainer(args, model, optimizer1, optimizer2, optimizer3, epoch):
    loss_epoch = []
    model.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()

    gnn_loss, pre_out, street_embedding = model()
    loss_epoch.append(gnn_loss.item())
    loss = gnn_loss

    loss.backward()

    optimizer1.step()
    optimizer2.step()
    optimizer3.step()

    if epoch % 10 == 0:
        print(
            f"TrainEpoch [{epoch + 1}/{args.epochs}]\t loss_epoch_gnn:{np.mean(loss_epoch)}")
    return np.mean(loss_epoch), pre_out, street_embedding


def test(args, model, epoch):
    with torch.no_grad():
        model.eval()
        _, out, _ = model()
        if args.downstream == 'poi':
            acc, f1, mrr, _, _, _, num, pred_out = model.test(out)
            return acc, f1, mrr, 1, 1, 1, num, pred_out
        else:
            a1, a3, a5, a10, f1, mrr, num, pred_out = model.test(out)
            return a1, a3, a5, a10, f1, mrr, num, pred_out


np.random.seed(args.seed)
random.seed(args.seed + 1)
torch.manual_seed(args.seed + 2)
torch.cuda.manual_seed(args.seed + 3)
torch.backends.cudnn.deterministic = True

model = SV_GAT(args)

model = model.to(args.device)

opt1 = torch.optim.Adam(
    itertools.chain(model.attention_soft.parameters(),), lr=0.0005, weight_decay=1e-8)
if args.downstream == 'poi':
    opt3 = torch.optim.Adam(model.gat_poi.parameters(), lr=0.0005, weight_decay=5e-4)
    args.epochs=200
else:
    opt3 = torch.optim.Adam(model.gat.parameters(), lr=0.005, weight_decay=5e-4)
if args.mode != 'mean':
    opt2 = torch.optim.SGD(model.sv_agg.parameters(), lr=0.005, weight_decay=1e-4, momentum=0.9)
    t = 10  # warmup
    T = 800  # epochs - 10 为 cosine rate
    n_t = 0.5
    lf = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lf)

print(model)

for epoch in range(args.start_epoch, args.epochs):
    loss_epoch, pred_, street_embedding = trainer(args, model, opt1, opt2, opt3, epoch)
    scheduler.step()
    if epoch in range(args.print_epoch, args.epochs + 10, args.print_num):
        test(args, model, epoch)
    args.current_epoch += 1

test(args, model, epoch)

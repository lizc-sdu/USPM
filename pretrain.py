import os
import pickle
import torch.distributed as dist

from torch.utils.data import DataLoader, sampler, DistributedSampler
import pandas as pd
from torch import nn
import torch
import numpy as np
import argparse

from torchvision import models
from transformers import AutoTokenizer, AutoModel

from ImageModel import ImageModel
from data_util import *

parser = argparse.ArgumentParser()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser.add_argument('--device', type=str, default='cuda:4', help='gpu device ids')
parser.add_argument("--start_epoch", type=int, default=0, nargs="?", help="Start epoch")
parser.add_argument("--current_epoch", type=int, default=0, nargs="?", help="Current epoch")
parser.add_argument("--global_step", type=int, default=0, nargs="?", help="global_step")
parser.add_argument("--epochs", type=int, default=100, nargs="?", help="Epochs")
parser.add_argument("--batch_size", type=int, default=4, nargs="?", help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0003, nargs="?", help="Learning rate.")
parser.add_argument("--seed", type=int, default=42, nargs="?", help="random seed.")
parser.add_argument("--model_path", type=str, default="model/")
parser.add_argument("--sv_path", type=str, default="../../../sv//")
args = parser.parse_args()

class SV_Emb:
    def __init__(self, sv_dim_0, sv_dim_1=768):
        self.sv_emb = torch.zeros(sv_dim_0, sv_dim_1).to(args.device)

    def emb_update(self, sv_embedding, sv_ids):
        self.sv_emb[sv_ids] = sv_embedding


def save_embedding(st_emb, epoch):
    sv_emb = st_emb.sv_emb
    with open('data/sv_list.pkl', 'rb') as f:
        sv_list = pickle.load(f)
    pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()
    sv_embedding = []
    for idx in range(5458):
        sv_emb_list = []
        for img_file_name in sv_list[idx]:
            emb = sv_emb[pic2id[img_file_name]]
            sv_emb_list.append(emb)
        sv_ = torch.stack(sv_emb_list, 0)
        sv_embedding.append(sv_)
    sv = []
    for item in sv_embedding:
        for i in item:
            sv.append(i)
    a = torch.stack(sv, 0)

    torch.save(a, 'embeddings/sv_huang117144_0201_{}.pt'.format(epoch))
    torch.save(sv_emb, 'embeddings/sv_huang57396_0201_{}.pt'.format(epoch))


def trainer(args, model, train_loader, optimizer, epoch):
    loss_epoch_con = []
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        sv, length, ids, sv_ids = batch[0].to(args.device), batch[1], batch[2], batch[3]
        loss_context = model(sv, length)
        loss = loss_context
        loss.backward()
        optimizer.step()

        loss_epoch_con.append(loss_context.item())

        if step % 20 == 0:
            print('===================')
            print(
                f"TrainStep [{step}/{len(train_loader)}]\t con_loss_epoch:{np.mean(loss_epoch_con)}")

    print(
        f"TrainEpoch [{epoch + 1}/{args.epochs}]\t con_loss_epoch:{np.mean(loss_epoch_con)}")
    # return np.mean(loss_epoch_au), np.mean(loss_epoch_con)


def tester(args, model, train_loader, st_emb):
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(train_loader):
            sv, length, ids, sv_ids = batch[0].to(args.device), batch[1], batch[2], batch[3]
            sv_emb_ = model.get_feature(sv)
            st_emb.emb_update(sv_emb_, sv_ids)

            args.global_step += 1


def collate_fn(batch):
    sv = []
    length = []
    ids = []
    sv_ids = []

    for unit in batch:
        sv.append(unit[0])
        length.append(unit[1])
        ids.append(unit[2])
        for u in unit[3]:
            sv_ids.append(u)

    sv_ = torch.cat(sv, dim=0)

    return sv_, length, ids, sv_ids


def collate_fn2(batch):
    sv = []
    ids = []
    sv_ids = []

    for unit in batch:
        sv.append(unit[0])
        ids.append(unit[1])
        for u in unit[2]:
            sv_ids.append(u)

    sv_ = torch.cat(sv, dim=0)

    return sv_, ids, sv_ids


torch.manual_seed(args.seed)
np.random.seed(args.seed)

train_dataset = street_dataset(args.sv_path)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)

test_dataset = region_dataset_test(args.sv_path)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                         collate_fn=collate_fn2, drop_last=False)

arch = 'resnet18'
model_place365_resnet18_path = "model_pretrain/resnet18_places365.pth.tar"
resnet18_pretrain_sv = torch.load(model_place365_resnet18_path, map_location=args.device)
sv_encoder = models.__dict__[arch](num_classes=365)
state_dict = {str.replace(k, 'module.', ''): v for k, v in resnet18_pretrain_sv['state_dict'].items()}
sv_encoder.load_state_dict(state_dict)
sv_encoder.fc = nn.Linear(512, 768)

# initialize model
model = ImageModel(sv_encoder, args)
model.to(args.device)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(args.start_epoch, args.epochs):
    trainer(args, model, train_loader, opt, epoch)
    if epoch in range(20, args.epochs + 40, 20):
        out = os.path.join(args.model_path, "checkpoint_{}_pretrain.tar".format(args.current_epoch))
        torch.save(model.state_dict(), out)

        sv_emb = SV_Emb(sv_dim_0=57396, sv_dim_1=768)
        tester(args, model, test_loader, sv_emb)
        save_embedding(sv_emb, epoch)
    args.current_epoch += 1

out = os.path.join(args.model_path, "checkpoint_{}_pretrain.tar".format(args.current_epoch))
torch.save(model.state_dict(), out)

sv_emb = SV_Emb(sv_dim_0=57396, sv_dim_1=768)
tester(args, model, test_loader, sv_emb)
save_embedding(sv_emb, args.current_epoch)


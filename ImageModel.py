import numpy as np
import torch
import torch.nn as nn
import torchvision
import collections
import torch.nn.functional as F
from transformers import AutoModel


def device_as(t1, t2):
    return t1.to(t2.device)


class InfoNCE(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        # self.batch_size = batch_size
        self.temperature = temperature

    @staticmethod
    def compute_batch_sim(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, polyline_embs, c_embs):
        b_size = polyline_embs.shape[0]
        polyline_norm = F.normalize(polyline_embs, p=2, dim=1)
        c_norm = F.normalize(c_embs, p=2, dim=1)
        mask = (~torch.eye(b_size * 2, b_size * 2, dtype=bool)).float()

        similarity_matrix = self.compute_batch_sim(polyline_norm, c_norm)

        sim_ij = torch.diag(similarity_matrix, b_size)
        sim_ji = torch.diag(similarity_matrix, -b_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * b_size)

        return loss


class ImageModel(nn.Module):
    def __init__(self, sv_encoder, args):
        super(ImageModel, self).__init__()
        self.args = args

        self.sv_encoder = sv_encoder

        self.projector_sv = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.info_nce = InfoNCE(temperature=0.1)

    def forward(self, sv, length):
        sv_resnet = self.sv_encoder(sv)
        sv_features = self.projector_sv(sv_resnet)
        street_list = list(torch.split(sv_features, length, dim=0))

        sv = []
        for idx in range(len(street_list)):
            sv.append(torch.mean(street_list[idx], dim=0))
        sv_mean = torch.stack(sv)

        context_emb = sv_mean.repeat_interleave(torch.tensor(length).to(self.args.device), dim=0)

        context_loss = self.info_nce(sv_features, context_emb)

        return context_loss

    def get_feature(self, sv):
        with torch.no_grad():
            sv_resnet = self.sv_encoder(sv)

        return sv_resnet

import numpy as np
import torch
import torch.nn as nn
import torchvision
import collections
import torch.nn.functional as F
from transformers import AutoModel


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ImageModel2(nn.Module):
    def __init__(self, sv_encoder, scn_encoder, args):
        super(ImageModel2, self).__init__()
        self.args = args

        self.sv_encoder = sv_encoder
        self.scn_bert = scn_encoder
        self.scn_bert.training = True

        self.projector_sv = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.projector_scn = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.info_nce = InfoNCE(temperature=0.1)

    def load_pretrain(self, pre_s_dict):
        s_dict = self.sv_encoder.state_dict()
        pre_s_dict.pop('projector.0.weight')
        pre_s_dict.pop('projector.2.weight')
        missing_keys = []
        new_state_dict = collections.OrderedDict()
        for key in s_dict.keys():
            simclr_key = 'encoder.' + key
            if simclr_key in pre_s_dict.keys():
                new_state_dict[key] = pre_s_dict[simclr_key]
            else:
                new_state_dict[key] = s_dict[key]
                missing_keys.append(key)
        print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
        # load new s_dict
        self.sv_encoder.load_state_dict(new_state_dict)

    def forward(self, sv, length):
        # print(f'batch_num = {len(sv)}')
        # sv = sv.reshape(sv.size(0)*sv.size(1), 3, 224, 224)  # 40*4*3*224*224 batchsize / 4 image for street/ 3 224 224

        sv_resnet = self.sv_encoder(sv)  # 3*224*224-->512
        sv_features = self.projector_sv(sv_resnet)
        street_list = list(torch.split(sv_features, length, dim=0))

        sv = []
        for idx in range(len(street_list)):
            sv.append(torch.mean(street_list[idx], dim=0))
        sv_mean = torch.stack(sv)  # 这里用的mean方法
        # street_emb = self.soft_attention(street_list)

        context_emb = sv_mean.repeat_interleave(torch.tensor(length).to(self.args.device), dim=0)

        # context_loss = contrastive_learning(sv_features, context_emb, length)
        context_loss = self.info_nce(sv_features, context_emb)

        return context_loss

    def get_feature(self, sv):
        with torch.no_grad():
            sv_resnet = self.sv_encoder(sv)

        return sv_resnet


def contrastive_learning(h_i, h_j, length):
    k, _ = h_i.size()

    temperature_f = 0.5
    sim = torch.matmul(h_i, h_j.T) / temperature_f
    sim_i_j = torch.diag(sim)
    positive_samples = sim_i_j.reshape(k, 1)
    mask = mask_correlated_samples(length)
    negative_samples = sim[mask].reshape(k, -1)

    labels = torch.zeros(k).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= k
    return loss


def mask_correlated_samples(length):
    k = sum(length)
    sample = torch.zeros(k)
    start = 0
    for i in range(len(length)):
        sample[start] = 1
        start = start + length[i]
    mask = torch.stack([sample for _ in range(k)], dim=0)
    line = 0
    start = 0
    for i in range(len(length)):
        mask[line:line + length[i], start] = 0
        start = start + length[i]
        line = line + length[i]
    mask = mask.bool()
    return mask


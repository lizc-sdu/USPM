import pickle
import random

import numpy as np
from torch.utils.data import Dataset
import augly.image as imaugs
import PIL.Image as Image
import torch
import torchvision
from transformers import AutoTokenizer


class street_dataset(Dataset):
    def __init__(self, sv_root_dir):
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.st_nid_list = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform

    def __len__(self):
        return len(self.st_nid_list)

    def __getitem__(self, street_idx):
        streetview_list = []
        sv_id = []

        if len(self.st_nid_list[street_idx]) > 16:
            img_list = random.sample(self.st_nid_list[street_idx], 16)
        else:
            img_list = self.st_nid_list[street_idx]

        for img_file_name in img_list:
            sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            streetview_list.append(image)

        length = len(streetview_list)

        streetview = torch.stack(streetview_list, 0)

        return streetview, length, street_idx, sv_id


class region_dataset_test(Dataset):
    def __init__(self, sv_root_dir):
        # self.street_idx = street_idx
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.st_nid_list = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform

    def __len__(self):
        return len(self.st_nid_list)

    def __getitem__(self, street_idx):
        streetview_list = []
        sv_id = []
        # scn_list = []

        img_list = self.st_nid_list[street_idx]

        for img_file_name in img_list:
            sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            streetview_list.append(image)

        streetview = torch.stack(streetview_list, 0)

        return streetview, street_idx, sv_id

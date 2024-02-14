import pickle
import random

import numpy as np
from torch.utils.data import Dataset
# import augly.image as imaugs
import PIL.Image as Image
import torch
import torchvision
from transformers import AutoTokenizer


class region_dataset(Dataset):
    def __init__(self, sv_root_dir):
        # self.street_idx = street_idx
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.sv_nid_list = pickle.load(f)
        with open('data/pic2scene.pickle', 'rb') as f:
            self.pic2scene = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform  #

        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        # self.encoded_scn = tokenizer(self.scene_list, padding=True, truncation=True, return_tensors='pt')

    def __len__(self):
        return len(self.sv_nid_list)

    def __getitem__(self, street_idx):
        streetview_list = []
        sv_id = []
        scn_list = []

        if len(self.sv_nid_list[street_idx]) > 16:
            img_list = random.sample(self.sv_nid_list[street_idx], 16)
        else:
            img_list = self.sv_nid_list[street_idx]

        for img_file_name in img_list:
            sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            streetview_list.append(image)
            scn_list.append(self.pic2scene[img_file_name])

        # scn_input = self.encoded_scn['input_ids'][street_idx]
        # scn_token = self.encoded_scn['token_type_ids'][street_idx]
        # scn_attention = self.encoded_scn['attention_mask'][street_idx]

        streetview = torch.stack(streetview_list, 0)
        scn = ".".join(scn_list)

        return streetview, scn, len(streetview_list), street_idx, sv_id


class street_dataset(Dataset):
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


class sv_dataset_level2(Dataset):
    def __init__(self, sv_root_dir):
        # self.street_idx = street_idx
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.st_nid_list = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()
        self.adj_streets = torch.load('data/adjacent_street.pt').tolist()

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
        adj_streetview_list = []

        adj = self.adj_streets[street_idx]

        if len(self.st_nid_list[street_idx]) > 16:
            img_list = random.sample(self.st_nid_list[street_idx], 16)
        else:
            img_list = self.st_nid_list[street_idx]


        if len(self.st_nid_list[adj]) > 16:
            adj_img_list = random.sample(self.st_nid_list[adj], 16)
        else:
            adj_img_list = self.st_nid_list[adj]


        for img_file_name in img_list:
            sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            streetview_list.append(image)


        for img_file_name in adj_img_list:
            # sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            adj_streetview_list.append(image)

        length1 = len(streetview_list)
        length2 = len(adj_streetview_list)

        streetview = torch.stack(streetview_list, 0)
        streetview_adj = torch.stack(adj_streetview_list, 0)

        return streetview, streetview_adj, length1, length2, street_idx, sv_id


class sv_dataset_multilevel(Dataset):
    def __init__(self, sv_root_dir, sv2st_dic):
        # self.street_idx = street_idx
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.sv_nid_list = pickle.load(f)
        self.sv2st_dic = sv2st_dic

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform

    def __len__(self):
        return len(self.sv_nid_list)

    def __getitem__(self, sv_idx):
        streetview_list = []
        streetview_aug_list = []
        streetview_near_list = []
        # image_pad = torch.zeros(3, 224, 224)

        img_file_name = self.sv_nid_list[sv_idx]
        nid = img_file_name.split('_')[0]
        near_file_name = random.choice(self.sv2st_dic[nid])

        image = Image.open(self.sv_root_dir + img_file_name)
        aug = imaugs.blur(image, radius=2.0)
        near = Image.open(self.sv_root_dir + near_file_name)
        if self.transform:
            image = self.transform(image)
            aug = self.transform(aug)
            near = self.transform(near)

            # streetview_list.append(image)
            # streetview_aug_list.append(aug)

            # if len(streetview_list) == 100:
            #     break
        # while len(streetview_list) < 12:
        #     streetview_list.append(image_pad)
        #
        # while len(streetview_aug_list) < 12:
        #     streetview_aug_list.append(image_pad)

        # streetview = torch.stack(streetview_list, 0)
        # streetview_aug = torch.stack(streetview_aug_list, 0)

        return image, aug, near, sv_idx


class sv_dataset_aug(Dataset):
    def __init__(self, sv_root_dir):
        self.sv_root_dir = sv_root_dir
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()
        with open('data/sv_list.pkl', 'rb') as f:
            sv_nid_list = pickle.load(f)

        li = []
        for st in sv_nid_list:
            for pi in st:
                li.append(pi)
        self.sv_list = list(set(li))

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform

    def __len__(self):
        return len(self.sv_list)

    def __getitem__(self, sv_idx):
        img_file_name = self.sv_list[sv_idx]
        sv_id = self.pic2id[img_file_name]

        image = Image.open(self.sv_root_dir + img_file_name)
        aug = imaugs.blur(image, radius=2.0)
        if self.transform:
            image = self.transform(image)
            aug = self.transform(aug)

        return image, aug, sv_id


class pretrained_dataset(Dataset):
    def __init__(self):
        with open('data/sv_list.pkl', 'rb') as f:
            self.sv_nid_list = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()
        self.sv_emb = torch.load('model/street_sv_emb_1218.pt')
        self.scn_emb = torch.load('model/street_scn_emb_1218.pt')

    def __len__(self):
        return len(self.sv_nid_list)

    def __getitem__(self, street_idx):
        sv_list = []

        for img_file_name in self.sv_nid_list[street_idx]:
            sv_embed = self.sv_emb[self.pic2id(img_file_name)]
            sv_list.append(sv_embed)

        scn_embedding = self.scn_emb[street_idx]
        sv_embedding = torch.stack(sv_list, 0)

        return sv_embedding, scn_embedding, len(sv_embedding), street_idx


class s_dataset(Dataset):
    def __init__(self):
        self.sv_nid_list = range(5458)

    def __len__(self):
        return len(self.sv_nid_list)

    def __getitem__(self, street_idx):
        return street_idx


class sv_dataset(Dataset):
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

        img_list = self.st_nid_list[street_idx]

        for img_file_name in img_list:
            sv_id.append(self.pic2id[img_file_name])
            image = Image.open(self.sv_root_dir + img_file_name)
            if self.transform:
                image = self.transform(image)

            streetview_list.append(image)

        streetview = torch.stack(streetview_list, 0)

        return streetview, street_idx, sv_id


class region_dataset_test(Dataset):
    def __init__(self, sv_root_dir):
        # self.street_idx = street_idx
        self.sv_root_dir = sv_root_dir
        with open('data/sv_list.pkl', 'rb') as f:
            self.st_nid_list = pickle.load(f)
        self.pic2id = np.load('data/pic2id.npy', allow_pickle='TRUE').item()
        with open('data/pic2scene.pickle', 'rb') as f:
            self.pic2scene = pickle.load(f)

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
            # scn_list.append(self.pic2scene[img_file_name])

        streetview = torch.stack(streetview_list, 0)
        # scn = ".".join(scn_list)

        return streetview, street_idx, sv_id

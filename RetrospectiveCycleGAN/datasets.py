'''
@saulzhang
The implementation code of dataset loader class in the the paper 
"Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
data: Nov,17,2019
'''

import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle

from torch.utils.data import Dataset
from PIL import Image

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# def video_load(video_path,nt=5):
#     video_info = []
#     video_folder = os.listdir(video_path)
#     for video_name in video_folder:
#         num_img = len(os.listdir(os.path.join(video_path,video_name)))
#         for j in range(num_img-nt+1):
#             index_set = []
#             for k in range(j, j + nt):
#                 index_set.append(os.path.join(video_path,os.path.join(video_name,str(k)+".jpg")))
#             video_info.append(index_set)
#     return video_info

class ImageTrainDataset(Dataset):
    def __init__(self, video_pkl_file, transforms_=None,nt=5):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        train_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(train_pkl_file)
        # self.data = video_load(video_path,nt)

        # if not os.path.exists("./dataset/pickle_data/train_data.pkl"):#引入该模块的意义在于使得test和train的划分可以持久化的保存下来，方便反复进行实验
        #     print("create folder './dataset/pickle_data'")
        #     os.makedirs("./dataset/pickle_data")
        #     train_dataset_output = open('./dataset/pickle_data/train_data.pkl', 'wb')
        #     pickle.dump(self.data, train_dataset_output)
        #     train_dataset_output.close()
        # else:
        #     train_pkl_file = open('./dataset/pickle_data/train_data.pkl', 'rb')
        #     self.data = pickle.load(train_pkl_file)

    def __getitem__(self, index):
        frame_seq = []
        Flip_p = 0.0#不翻转
        if random.random() < 0.3:#以0.3的概率对整个图片序列进行水平翻转
            Flip_p = 1.0#翻转
        for img_name in self.data[index]:
            img = Image.open(img_name)
            img = transforms.RandomHorizontalFlip(p=Flip_p)(img)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0)
        return frame_seq

    def __len__(self):
        return len(self.data)

class ImageTestDataset(Dataset):
    def __init__(self, video_pkl_file, transforms_=None,nt=5):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        # self.data = video_load(video_path,nt)
        train_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(train_pkl_file)
        # if not os.path.exists("./dataset/pickle_data/test_data.pkl"):#引入该模块的意义在于使得test和train的划分可以持久化的保存下来，方便反复进行实验
        #     if not os.path.exists("./dataset/pickle_data/"):
        #         print("create folder './dataset/pickle_data'")
        #         os.makedirs("./dataset/pickle_data")
        #     test_dataset_output = open('./dataset/pickle_data/test_data.pkl', 'wb')
        #     pickle.dump(self.data, test_dataset_output)
        #     test_dataset_output.close()
        # else:
        #     test_pkl_file = open('./dataset/pickle_data/test_data.pkl', 'rb')
        #     self.data = pickle.load(test_pkl_file)

    def __getitem__(self, index):
        frame_seq = []
        for img_name in self.data[index]:
            img = Image.open(img_name)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0)
        return frame_seq

    def __len__(self):
        return len(self.data)

'''
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models import *
from utils import *
from laplacian_of_guassian import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="./dataset", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height") #当图片的宽和高设置为128的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence")

opt = parser.parse_args()##返回一个命名空间,如果想要使用变量,可用args.attr
print(opt)
# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),      #如果要水平翻转的话，整个视频都得翻转
    transforms.ToTensor(),#Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                          #Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                          #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
# Training data loader
dataloader = DataLoader(
    ImageTrainDataset(opt.dataset_name+"/train", transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageTestDataset(opt.dataset_name+"/test", transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
for i, frame_seq in enumerate(dataloader):
'''

'''
@saulzhang
The implementation code of test the testing dataset in the paper 
"Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
date: Nov,12,2019
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
import metric

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models import *
from datasets import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--img_height", type=int, default=128, help="size of image height") #当图片的宽和高设置为128的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=160, help="size of image width")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence")

opt = parser.parse_args()##返回一个命名空间,如果想要使用变量,可用args.attr
print(opt)

cuda = torch.cuda.is_available()

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageTestDataset('./dataset/KITTI/pkl/test_data.pkl', transforms_=transforms_,nt=opt.sequence_len),
    #ImageTestDataset('./dataset/test_data_caltech.pkl', transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)

input_shape = (opt.batch_size, opt.sequence_len, opt.channels, opt.img_height, opt.img_width)
# print("=====output of the Generator=====")
G = GeneratorResNet(input_shape, opt.n_residual_blocks)
G = torch.nn.DataParallel(G, device_ids=range(torch.cuda.device_count()))

if cuda:
    G = G.cuda()

if cuda:
    G.load_state_dict(torch.load("saved_models_KITTI/G_%d.pth" %  opt.epoch))
else:
    G.load_state_dict(torch.load("saved_models_KITTI/G_%d.pth" %  opt.epoch, map_location='cpu'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

count = 0
tatal_PSNR = 0
total_SSIM = 0
total_MSE = 0

#calculate the mse,psnr ans ssim over the testing data.
for i, frame_seq in enumerate(val_dataloader):

    frame_seq = frame_seq.type(Tensor)
    real_A = Variable(frame_seq[:,-1,...]) #[bs,1,c,h,w]
    input_A = Variable(frame_seq[:,:-1,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
    A1 = G(input_A)
    count += 1
    psnr = metric.PSNR(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    import skimage
    #ssim = metric.ssim(real_A.detach(),A1.detach(), data_range=1, size_average=False, K=(0.01, 0.4)).item()
    ssim = metric.SSIM(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    mse = metric.MSE(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())*1000
    tatal_PSNR += psnr
    total_SSIM += ssim
    total_MSE += mse
    print("[{}/{}][currunt psnr: {:.3f} average psnr: {:.3f}] [currunt ssim: {:.3f} average ssim: {:.3f}] \
[currunt mse: {:.3f} average mse: {:.3f}]".format(count,len(val_dataloader),psnr,tatal_PSNR/count,ssim,total_SSIM/count,mse,total_MSE/count))

print("Epoch: {} PSNR={}, SSIM={}, MSE={}".format(opt.epoch,tatal_PSNR/count,total_SSIM/count,total_MSE/count))

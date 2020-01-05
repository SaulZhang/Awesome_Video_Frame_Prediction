'''
@saulzhang
The implementation code of the paper "Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
data: Nov,12,2019
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
from laplacian_of_guassian import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--train_data", type=str, default="./dataset/KITTI/pkl/train_data.pkl", help="the path of pickle file about train data")
parser.add_argument("--val_data", type=str, default="./dataset/KITTI/pkl/val_data.pkl", help="the path of pickle file about validation data")
parser.add_argument("--test_data", type=str, default="./dataset/test_data_caltech.pkl", help="the path of pickle file about testing data")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height") #当图片的宽和高设置为256的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_LoG", type=float, default=0.005, help="cycle loss weight")
parser.add_argument("--lambda_frame_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--lambda_seq_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence(n+1)")
parser.add_argument("--save_model_path", type=str, default="saved_models/KITTI/", help="the path of saving the models")
parser.add_argument("--save_image_path", type=str, default="saved_images/KITTI/", help="the path of saving the images")
parser.add_argument("--log_file", type=str, default="log_file.txt", help="the logging info of training")


opt = parser.parse_args()##返回一个命名空间,如果想要使用变量,可用args.attr
print(opt)

# Create sample and checkpoint directories
os.makedirs(opt.save_image_path, exist_ok=True)
os.makedirs(opt.save_model_path, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_Limage = torch.nn.L1Loss()
cuda = torch.cuda.is_available()

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    transforms.ToTensor(),#Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                          #Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                          #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#[0,1] -> [-1,1]
]

# Training data loader
dataloader = DataLoader(
    ImageTrainDataset(opt.train_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# val data loader
val_dataloader = DataLoader(
    ImageTestDataset(opt.val_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# test data loader
test_dataloader = DataLoader(
    ImageTestDataset(opt.test_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

input_shape = (opt.batch_size, opt.sequence_len, opt.channels, opt.img_height, opt.img_width)
G = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = DiscriminatorA(input_shape)
D_B = DiscriminatorB(input_shape)

Laplacian = Laplacian()

if cuda:
    G = torch.nn.DataParallel(G, device_ids=range(torch.cuda.device_count()))
    G = G.cuda()
    D_A = torch.nn.DataParallel(D_A, device_ids=range(torch.cuda.device_count()))
    D_A = D_A.cuda()
    D_B = torch.nn.DataParallel(D_B, device_ids=range(torch.cuda.device_count()))
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_Limage.cuda()
    Laplacian = Laplacian.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load(opt.save_model_path+"G_%d.pth" %  opt.epoch))
    D_A.load_state_dict(torch.load(opt.save_model_path+"D_A_%d.pth" %  opt.epoch))
    D_B.load_state_dict(torch.load(opt.save_model_path+"D_B_%d.pth" %  opt.epoch))
else:
    # Initialize weights
    G.apply(weights_init_normal)#apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step# lr_lambda为操作学习率的函数
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#保存中间的训练结果
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(dataloader))
    imgs = imgs.type(Tensor)
    input_A = imgs[:,:-1,...]
    input_A = input_A.view((imgs.size(0),-1,)+imgs.size()[3:])
    G.eval()
    real_A = Variable(imgs[:,-1:,...])
    A1 = G(input_A)
    frames = torch.cat((imgs[0,:-1,],A1[0].unsqueeze(0),imgs[0]), 0)
    image_grid = make_grid(frames,nrow=opt.sequence_len,normalize=False)
    save_image(image_grid, opt.save_image_path+"fake_%s.png" % (batches_done), normalize=False)

def ReverseSeq(Seq):
    length = Seq.size(1)
    return torch.cat([Seq[:,i-2:i+1,...] for i in range(length-1,-1,-3)],1)


count = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, frame_seq in enumerate(dataloader):

        count += 1
        frame_seq = frame_seq.type(Tensor)

        real_A = Variable(frame_seq[:,-1,...]) #[bs,1,c,h,w]
        input_A = Variable(frame_seq[:,:-1,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        real_B = Variable(frame_seq[:,0,...]) #[bs,1,c,h,w]
        input_B_ = Variable(frame_seq[:,1:,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        input_B = ReverseSeq(input_B_)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((frame_seq.size(0), *D_A.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((frame_seq.size(0), *D_A.module.output_shape))), requires_grad=False)

        #------------------------
        #  Train Generator
        #------------------------
        G.train()
        optimizer_G.zero_grad()#梯度清零

        #L_Image loss which minimize the L1 Distance between the image pair
        A1 = G(input_A) # x^'_{n}
        B1 = G(input_B) # x^'_{m}
        input_A_A1_ = torch.cat((input_A[:,3:,...],A1),1)
        input_A_A1 = ReverseSeq(input_A_A1_)
        input_B_B1 = torch.cat((B1,input_A[:,3:,...]),1)
        A11 = G(input_B_B1)# x^''_{n}
        B11 = G(input_A_A1)# x^''_{m}

        loss_A_A1 = criterion_Limage(real_A,A1)
        loss_A_A11 = criterion_Limage(real_A,A11)
        loss_A1_A11 = criterion_Limage(A1,A11)
        loss_B_B1 = criterion_Limage(real_B,B1)
        loss_B_B11 = criterion_Limage(real_B,B11)
        loss_B1_B11 = criterion_Limage(B1,B11)

        loss_Image = (loss_A_A1 + loss_A_A11 + loss_A1_A11 + loss_B_B1 + loss_B_B11 + loss_B1_B11) / 6
        #L_LoG loss 
        L_LoG_A_A1 = criterion_Limage(Laplacian(real_A),Laplacian(A1))
        L_LoG_A_A11 = criterion_Limage(Laplacian(real_A),Laplacian(A11))
        L_LoG_A1_A11 = criterion_Limage(Laplacian(A1),Laplacian(A11))
        L_LoG_B_B1 = criterion_Limage(Laplacian(real_B),Laplacian(B1))
        L_LoG_B_B11 = criterion_Limage(Laplacian(real_B),Laplacian(B11))
        L_LoG_B1_B11 = criterion_Limage(Laplacian(B1),Laplacian(B11))

        loss_LoG = (L_LoG_A_A1 + L_LoG_A_A11 + L_LoG_A1_A11 + L_LoG_B_B1 + L_LoG_B_B11 + L_LoG_B1_B11) / 6

        #GAN frame Loss(Least Square Loss)
        loss_frame_GAN_A1  = criterion_GAN(D_A(A1),valid)# lead the synthetic frame become similiar to the real frame
        loss_frame_GAN_B1  = criterion_GAN(D_A(B1),valid)
        loss_frame_GAN_A11 = criterion_GAN(D_A(A11),valid)
        loss_frame_GAN_B11 = criterion_GAN(D_A(B11),valid)
        #Total frame loss
        loss_frame_GAN = (loss_frame_GAN_A1 + loss_frame_GAN_B1 + loss_frame_GAN_A11 + loss_frame_GAN_B11) / 4
        # print("Frame Loss Done")

        #GAN seq Loss 
        #four kinds of the synthetic frame sequence
        input_B1_A  = torch.cat((B1,input_A[:,3:,...],real_A),1)
        input_B11_A1 = torch.cat((B11,input_A[:,3:,...],A1),1)
        input_B_A1 = torch.cat((real_B,input_A[:,3:,...],A1),1)
        input_B1_A11 = torch.cat((B1,input_A[:,3:,...],A11),1)
        loss_seq_GAN_B1_A = criterion_GAN(D_B(input_B1_A),valid)
        loss_seq_GAN_B11_A1 = criterion_GAN(D_B(input_B11_A1),valid)
        loss_seq_GAN_B_A1 = criterion_GAN(D_B(input_B_A1),valid)
        loss_seq_GAN_B1_A11 = criterion_GAN(D_B(input_B1_A11),valid)
        # Total seq loss
        loss_seq_GAN = (loss_seq_GAN_B1_A + loss_seq_GAN_B11_A1 + loss_seq_GAN_B_A1 + loss_seq_GAN_B1_A11) / 4

        # Total GAN loss
        total_loss_GAN = loss_Image + opt.lambda_LoG*loss_LoG + opt.lambda_frame_GAN *loss_frame_GAN + opt.lambda_seq_GAN*loss_seq_GAN
        total_loss_GAN.backward() #反向传播，对各个变量求导
        optimizer_G.step() # 更新

        #------------------------
        #  Train Discriminator A
        #------------------------
        optimizer_D_A.zero_grad()
        # Real loss
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        loss_real_B = criterion_GAN(D_A(real_B), valid)
        # Fake loss
        loss_fake_A1 = criterion_GAN(D_A(A1.detach()), fake)#detach() 将Variable从计算图中抽离出来，进行梯度阶段。注意如果
                                                            #这里没有加上detach()函数的话会导致梯度传到G，而G的计算图已经被释放会报错
        loss_fake_A11 = criterion_GAN(D_A(A11.detach()), fake)
        loss_fake_B1 = criterion_GAN(D_A(B1.detach()), fake)
        loss_fake_B11 = criterion_GAN(D_A(B11.detach()), fake)
        # Total loss
        loss_D_A = (loss_real_A + loss_real_B + loss_fake_A1 + loss_fake_A11 +loss_fake_B1 + loss_fake_B11 ) / 6
        loss_D_A.backward()#retain_graph=True 
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        #real loss
        loss_real = criterion_GAN(D_B(Variable(frame_seq.view((frame_seq.size(0),-1)+frame_seq.size()[3:]))), valid)
        #fake loss
        loss_fake_B1_A = criterion_GAN(D_B(input_B1_A.detach()),fake)
        loss_fake_B11_A1 = criterion_GAN(D_B(input_B11_A1.detach()),fake)
        loss_fake_B_A1 = criterion_GAN(D_B(input_B_A1.detach()),fake)
        loss_fake_B1_A11 = criterion_GAN(D_B(input_B1_A11.detach()),fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake_B1_A + loss_fake_B11_A1 + loss_fake_B_A1 + loss_fake_B1_A11) / 5
        loss_D_B.backward()
        optimizer_D_B.step()
        total_loss_D = (loss_D_A + loss_D_B)/2
        # --------------
        #  Log Progress
        # --------------
        batches_done = epoch*len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        # Print log
        if count % 100 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, img: %f, LoG: %f, adv_frame: %f, adv_seq: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    total_loss_D.item(),
                    total_loss_GAN.item(),
                    loss_Image.item(),
                    loss_LoG.item(), 
                    loss_frame_GAN.item(),
                    loss_seq_GAN.item(),
                    time_left,
                )
            )
        if count > opt.sample_interval/opt.batch_size:
            sample_images(batches_done)
            count = 0
    # Update learning rates linear decay each 100 epochs
    if epoch % 100 == 0:
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    if epoch % 5 == 0:
        num = 0
        tatal_PSNR = 0
        total_SSIM = 0
        total_MSE = 0
        for i, frame_seq_test in enumerate(test_dataloader):
            frame_seq_test = frame_seq_test.type(Tensor)
            real_A_test = Variable(frame_seq_test[:,-1,...]) #[bs,1,c,h,w]
            input_A = Variable(frame_seq_test[:,:-1,...].view((frame_seq_test.size(0),-1)+frame_seq_test.size()[3:]))
            A1 = G(input_A)
            num += 1
            psnr = metric.PSNR(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            ssim = metric.SSIM(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            mse = metric.MSE(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())*1000
            tatal_PSNR += psnr
            total_SSIM += ssim
            total_MSE += mse
        testinfo = "Epoch: {} PSNR={}, SSIM={}, MSE={}\n".format(epoch,tatal_PSNR/num,total_SSIM/num,total_MSE/num)
        with open(opt.log_file, 'a+') as f:
              f.write(testinfo)

    #save the model
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(),   opt.save_model_path+"G_%d.pth" % (epoch))
        torch.save(D_A.state_dict(), opt.save_model_path+"D_A_%d.pth" % (epoch))
        torch.save(D_B.state_dict(), opt.save_model_path+"D_B_%d.pth" % (epoch))

'''
@saulzhang
The implementation code of LoG filter algorithm by convolution filter approximately
data: Nov,16,2019
'''
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from math import exp
from PIL import Image
from torchvision.utils import save_image, make_grid

class Laplacian(nn.Module):

    def __init__(self):
        super(Laplacian, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.window=torch.Tensor([[[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]])
        self.window_size=9
        self.save=False
    def forward(self, x):
        channel = x.size()[1]
        self.window = Variable(self.window.expand(channel, 1, self.window_size, self.window_size).contiguous())
        # Max pooling over a (2, 2) window\
        if torch.cuda.is_available():self.window=self.window.cuda()
        x = F.conv2d(x, self.window, padding = self.window_size//2, groups = channel)
        return x
'''
window  = torch.Tensor([[[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]])
window_size = 9
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

laplacian = Laplacian()

if cuda:
    laplacian = laplacian.cuda()

'''
def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)    
    return (data - min)/(max-min)


#输入为采用Image打开的图片 [h,w]
def LoG_Test4Img(img,window,window_size=9,mode="RGB",save=False):
    img1_array = np.array(img,dtype=np.float32)#Image -> array
    img1_tensor = torch.from_numpy(img1_array)# array -> tensor
    # 处理不同通道数的数据
    if mode == 'L':
        img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0)#h,w -> n,c,h,w 
    else:#RGB or RGBA
        img1_tensor = img1_tensor.permute(2,0,1)# h,w,c -> c,h,w 
        img1_tensor = img1_tensor.unsqueeze(0)#c,h,w -> n,c,h,w 
    channel = img1_tensor.size()[1]
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    output = F.conv2d(img1_tensor, window, padding = window_size//2, groups = channel)
    output = minmaxscaler(output)# 归一化到0~1之间
    if save:
        if (channel==4):
            save_image(output, "LoG_output.png", normalize=False)
        else:
            save_image(output, "LoG_output.jpg", normalize=False)

    return output

#输入为torch.Tensor类型的单张图片数据 [c,h,w]
def LoG(img,window=torch.Tensor([[[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]]),window_size=9,save=False):
    img = img.type(Tensor)
    img = img.unsqueeze(0)#c,h,w -> n,c,h,w 
    # channel = img.size()[1]
    # window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    # if cuda:
    #     output = F.conv2d(img, window, padding = window_size//2, groups = channel).cuda()
    # else:
    #     output = F.conv2d(img, window, padding = window_size//2, groups = channel)
    output = laplacian(img)
    output = minmaxscaler(output)# 归一化到0~1之间

    return output

'''
if __name__ == '__main__':

    img = Image.open("./metric_test_image/wdg4.gif")
    img = img.convert('L') # 灰度化
    LoG(img,window,window_size,img.mode)
'''

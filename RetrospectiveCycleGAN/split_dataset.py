import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle

from torch.utils.data import Dataset
from PIL import Image

def video_load(video_path,nt=5):
    video_info = []
    video_folder = os.listdir(video_path)
    cnt=0
    for video_name in video_folder:
        cnt += 1
        print(cnt)
        img_list = os.listdir(os.path.join(video_path,video_name))
        num_img = len(img_list)
        for j in range(num_img-nt+1):
            index_set = []
            for k in range(j, j + nt):
                index_set.append(os.path.join(os.path.join(video_path,video_name),img_list[k]))
            video_info.append(index_set)
    return video_info

def split_dataset4UCF101(video_path):#video_path = "/home/share2/ucf-data/jpegs_256"

    data_index = video_load(video_path)

    random.shuffle(data_index)

    trian_data = data_index[:int(0.9*len(data_index))]
    test_data = data_index[int(0.9*len(data_index)):]

    if not os.path.exists("./dataset/train"):
        os.mkdir("./dataset/train")

    train_output = open('./dataset/train/train_data.pkl', 'wb')
    pickle.dump(trian_data, train_output)


    if not os.path.exists("./dataset/test"):
        os.mkdir("./dataset/test")

    test_output = open('./dataset/test/test_data.pkl', 'wb')
    pickle.dump(test_data, test_output)

def split_dataset4KITTI(dataset_path):
    categories = ['city', 'residential', 'road']

    # Recordings used for validation and testing.
    # Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
    val_recordings = [('city', '2011_09_26_drive_0005_sync')]
    test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]
    #above partition come from the previous paper.
    
    #create the folder to save the pickle file..
    if not os.path.exists("./dataset/KITTI/pkl/"):
        os.mkdir("./dataset/KITTI/pkl/")

    train_video_path = os.path.join(dataset_path,"train")

    train_data = video_load(train_video_path)
    train_output = open('./dataset/KITTI/pkl/train_data.pkl', 'wb')
    pickle.dump(train_data, train_output)


    val_video_path = os.path.join(dataset_path,"val")
    val_data = video_load(val_video_path)
    val_output = open('./dataset/KITTI/pkl/val_data.pkl', 'wb')
    pickle.dump(val_data, val_output)


    test_video_path = os.path.join(dataset_path,"test")
    test_data = video_load(test_video_path)
    test_output = open('./dataset/KITTI/pkl/test_data.pkl', 'wb')
    pickle.dump(test_data, test_output)

def split_dataset4Caltech(video_path):
    video_info = []
    video_folder = os.listdir(video_path)
    cnt=0
    nt=5
    for video_name in video_folder:
        if int(video_name[3:5])<= 5:
            continue
        cnt += 1
        print(cnt)
        img_list = os.listdir(os.path.join(video_path,video_name))
        num_img = len(img_list)
        for j in range(num_img-nt+1):
            index_set = []
            for k in range(j, j + nt):
                index_set.append(os.path.join(os.path.join(video_path,video_name),img_list[k]))
            video_info.append(index_set)

    # random.shuffle(data_index)
    test_data = video_info

    test_output = open('./dataset/test_data_caltech.pkl', 'wb')
    pickle.dump(test_data, test_output)


if __name__ == '__main__':
    split_dataset4Caltech("./dataset/caltech/")

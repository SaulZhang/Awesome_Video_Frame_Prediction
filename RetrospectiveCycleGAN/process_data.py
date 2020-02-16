'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os,sys
import requests
from bs4 import BeautifulSoup
import urllib
import numpy as np
from imageio import imread
import hickle as hkl
import scipy
import scipy.misc
from PIL import Image
# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
DATA_DIR = '"G:/dataset/KITTI_test/raw/city"'

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
WEIGHTS_DIR = './model_data_keras2/'

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = './kitti_results/'



desired_im_sz = (128, 160)
categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]
                             
# if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Download raw zip files by scraping KITTI website
def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        # print(soup)
        drive_list = soup.find_all("h3")
        print(drive_list)
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
        print("Downloading set: " + c)
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)
        for i, d in enumerate(drive_list):
            print(str(i+1) + '/' + str(len(drive_list)) + ": " + d)
            url = "http://kitti.is.tue.mpg.de/kitti/raw_data/" + d + "/" + d + "_sync.zip"
            # print(url)
            # urllib.request.urlretrieve(url, filename=c_dir + d + "_sync.zip")


# unzip images
def extract_data():
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
        _, _, zip_files = os.walk(c_dir).__next__()
        for f in zip_files:
            print('unpacking: ' + f)
            spec_folder = f[:10] + '/' + f[:-4] + '/image_03/data*'
            command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4]
            os.system(command)


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    while True:
        try:
            for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
                c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
                print(c_dir)
                _, folders, _ = os.walk(c_dir).__next__()

                splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]
                print(splits['train'])

            for split in splits:
                print("@@@@",split)
                im_list = []
                source_list = []  # corresponds to recording that image came from
                for category, folder in splits[split]:
                    im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')
                    _, _, files = os.walk(im_dir).__next__()
                    im_list += [im_dir + f for f in sorted(files)]
                    source_list += [category + '-' + folder] * len(files)
                print(im_list)

                print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
                X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
                for i, im_file in enumerate(im_list):
                    im = imread(im_file)
                    X[i] = process_im(im, desired_im_sz)

                hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
                hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

        except StopIteration:
            sys.exit()

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = scipy.misc.imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = (im.shape[1] - desired_sz[1]) // 2
    im = im[:, d:d+desired_sz[1]]
    return im


def process_Caltech():
    DATA_DIR = "D:/caltech/test/images"
    img_list = os.listdir(DATA_DIR)
    for img in img_list:
        img_path = os.path.join(DATA_DIR,img)
        im = imread(img_path)
        print("Process: ",img_path)
        im = process_im(im, desired_im_sz)
        im = Image.fromarray(im)
        im.save(img_path)

if __name__ == '__main__':
    process_Caltech()

# coding:utf-8
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import pdb
import os
import cv2

class lmdbDataset(Dataset):

    def __init__(self, roots=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test'):
        self.envs = []
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        for i in range(0,len(roots)):
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (root))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.envs.append(env)

        if ratio != None:
            assert len(roots) == len(ratio) ,'length of ratio must equal to length of roots!'
            for i in range(0,len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0,len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))

        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        self.target_ratio = img_width / float(img_width)

    def __fromwhich__(self ):
        rd = random.random()
        total = 0
        for i in range(0,len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        cur_ratio = img.size[0] / float(img.size[1])

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0])/2)
        start_y = int((mask_width - img.shape[1])/2)
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[start_x : start_x + img.shape[0], start_y : start_y + img.shape[1]] = img
        img = mask        
        return img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))
            if len(label) > 24 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
            sample = {'image': img, 'label': label}
            return sample
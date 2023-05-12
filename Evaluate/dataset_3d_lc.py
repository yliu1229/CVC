from torch.utils import data
from torchvision import transforms
import os
import sys
import pandas as pd

sys.path.append('../Utils')
from Utils.augmentation import *


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 num_seq=8,
                 downsample=3,
                 epsilon=5,
                 which_split=1):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = '../ProcessData/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
        elif (mode == 'val') or (mode == 'test'):
            split = '../ProcessData/data/ucf101/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../ProcessData/data/ucf101', 'classInd.txt')
        # action_file = os.path.join('../ProcessData/data/ucf101', '1.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3)

    def idx_sampler(self, vlen):
        '''sample index from a video'''
        if vlen-self.num_seq*self.downsample <= 0: return None
        n = 1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample)  # all possible frames with downsampling
            return seq_idx_block
        start_idx = np.random.choice(range(int(vlen) - self.num_seq * self.downsample), n)
        # seq_idx is of (num_seq, 1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        idx_block = self.idx_sampler(vlen)

        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, 1)
            idx_block = idx_block.reshape(self.num_seq)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)

        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            # SL = the num of downsampled frames
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.num_seq <= SL:
                clips.append(t_seq[i:i+self.num_seq, :])
                # i += self.num_seq
                # for small GPU, use a bigger strip as self.num_seq*2
                i += self.num_seq*2
            # t_seq of (B, num_seq, C, dim, dim)
            t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, C, H, W)

        try:
            vname = vpath.split('/')[-2].split('_')[1]
            vid = self.encode_action(vname)
        except:
            print('error path: ', vpath)
            #vname = vpath.split('/')[-3].split('_')[1]
            #vid = self.encode_action(vname)

        label = torch.LongTensor([vid])

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class HMDB51_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 num_seq=8,
                 downsample=3,
                 epsilon=5,
                 which_split=1):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = '../ProcessData/data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../ProcessData/data/hmdb51/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../ProcessData/data/hmdb51', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.4)
        # shuffle not required

    def idx_sampler(self, vlen):
        '''sample index from a video'''
        if vlen-self.num_seq*self.downsample <= 0: return None
        n = 1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return seq_idx_block
        start_idx = np.random.choice(range(int(vlen)-self.num_seq*self.downsample), n)
        # seq_idx is of (num_seq, 1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        idx_block = self.idx_sampler(vlen)
        if idx_block is None: print(vpath)

        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, 1)
            idx_block = idx_block.reshape(self.num_seq)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform

        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            print(t_seq.size())
            import ipdb; ipdb.set_trace()

        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.num_seq <= SL:
                clips.append(t_seq[i:i+self.num_seq, :])
                # i += self.seq_len//2
                i += self.num_seq
            # t_seq of (B, num_seq, C, H, W)
            t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, C, H, W)

        try:
            vname = vpath.split('/')[4]
            vid = self.encode_action(vname)
        except:
            print('error path: ', vpath)
            # vname = vpath.split('/')[-3]
            # vid = self.encode_action(vname)

        label = torch.LongTensor([vid])

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


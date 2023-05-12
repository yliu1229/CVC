from torch.utils import data
from torchvision import transforms
import os
import sys
import decord
import pandas as pd

sys.path.append('../Utils')
from Utils.augmentation import *


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Kinetics(data.Dataset):
    """Load Kinetics400/600.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    num_seq : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    downsample : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default True.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 mode='train',
                 video_ext='mp4',
                 num_segments=1,
                 num_seq=8,
                 downsample=3,
                 transform=None,
                 transform_aug=None,
                 temporal_jitter=False,
                 video_loader=True):

        super(Kinetics, self).__init__()
        self.num_segments = num_segments
        self.num_seq = num_seq
        self.downsample = downsample
        self.skip_length = self.num_seq * self.downsample
        self.temporal_jitter = temporal_jitter
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.transform = transform
        self.transform_aug = transform_aug

        if mode == 'train':
            csv_path = 'C:/VideoData/Kinetics400/data_list/train_list.csv'
        elif (mode == 'val') or (mode == 'test'):  # val and test list are the same
            csv_path = 'C:/VideoData/Kinetics400/data_list/test_list.csv'
        else:
            raise ValueError('wrong mode')

        self.clips = self._make_dataset(csv_path)
        if len(self.clips) == 0:
            raise (RuntimeError("Found 0 video clips in subfolders of: \nCheck your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            try:
                decord_vr = decord.VideoReader(video_name, num_threads=1)
                duration = len(decord_vr)
            except:
                raise RuntimeError('Error occurred in decord reading video {}.'.format(video_name))

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data = self.transform(images)  # T*C,H,W

        (C, H, W) = process_data[0].size()
        process_data = torch.stack(process_data, 0)
        process_data = process_data.view(self.num_seq, C, H, W)

        process_data_aug = None
        if self.transform_aug is not None:
            process_data_aug = self.transform_aug(images)
            (C, H, W) = process_data_aug[0].size()
            process_data_aug = torch.stack(process_data_aug, 0)
            process_data_aug = process_data_aug.view(self.num_seq, C, H, W)

        return process_data, process_data_aug

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, csv_path):
        if not os.path.exists(csv_path):
            raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (csv_path)))
        clips = []
        with open(csv_path) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(',')
                # line format: video_path, video_label
                if len(line_info) < 2:
                    raise (RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.downsample, size=self.skip_length // self.downsample)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.downsample, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.downsample)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.downsample < duration:
                    offset += self.downsample
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError(
                'Error occurred in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 transform_aug=None,
                 num_seq=8,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.transform_aug = transform_aug
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '../ProcessData/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '../ProcessData/data/ucf101/test_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../ProcessData/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen):
        '''sample index from a video'''
        if vlen - self.num_seq * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(int(vlen) - self.num_seq * self.downsample), n)
        # seq_idx is of (num_seq, 1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        seq_idx = self.idx_sampler(vlen)
        if seq_idx is None: print(vpath)

        assert seq_idx.shape == (self.num_seq, 1)
        seq_idx = seq_idx.reshape(self.num_seq)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in seq_idx]
        t_seq = self.transform(seq)

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, C, H, W)

        t_aug_seq = None
        if self.transform_aug is not None:
            t_aug_seq = self.transform_aug(seq)  # apply augmentation
            (C, H, W) = t_aug_seq[0].size()
            t_aug_seq = torch.stack(t_aug_seq, 0)
            t_aug_seq = t_aug_seq.view(self.num_seq, C, H, W)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq, t_aug_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
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
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label

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
        start_idx = np.random.choice(range(int(vlen)-self.num_seq*self.downsample), n)
        # seq_idx is of (num_seq, 1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        idx_block = self.idx_sampler(vlen)
        if idx_block is None: print(vpath)

        assert idx_block.shape == (self.num_seq, 1)
        idx_block = idx_block.reshape(self.num_seq)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform

        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
            t_seq = t_seq.view(self.num_seq, C, H, W)
        except:
            print(t_seq.size())
            import ipdb; ipdb.set_trace()

        if self.return_label:
            try:
                vname = vpath.split('/')[4]
                vid = self.encode_action(vname)
            except:
                print('error path: ', vpath)
                # vname = vpath.split('/')[-3]
                # vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return  t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


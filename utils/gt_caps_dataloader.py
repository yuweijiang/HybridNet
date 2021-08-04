import os
import json
import torch
import random
import numpy as np
import h5py
from torch.utils.data import Dataset

class VideoDataset_mix40(Dataset):

    def get_cms_vocab_size(self):
        return len(self.get_cms_vocab())

    def get_cap_vocab_size(self):
        return len(self.get_cap_vocab())

    def get_cms_vocab(self):
        return self.cms_ix_to_word

    def get_cap_vocab(self):
        return self.cap_ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode='train'):
        super(VideoDataset_mix40, self).__init__()
        self.mode = mode

        self.captions = json.load(open(opt['caption_json']))
        cms_info = json.load(open(opt['info_json']))
        self.cms_ix_to_word = cms_info['ix_to_word']
        self.cms_word_to_ix = cms_info['word_to_ix']
        self.splits = cms_info['videos']

        # Load caption dictionary
        cap_info = json.load(open(opt['cap_info_json']))
        self.cap_ix_to_word = cap_info['ix_to_word']
        self.cap_word_to_ix = cap_info['word_to_ix']

        print('Caption vocab size is ', len(self.cap_ix_to_word))
        print('CMS vocab size is ', len(self.cms_ix_to_word))
        print('number of train videos: ', len(self.splits['train']))
        print('number of test videos: ', len(self.splits['test']))
        print('number of val videos: ', len(self.splits['val']))

        self.feats_dir = opt['feats_dir']
        self.cap_max_len = opt['cap_max_len']

        print('load feats from %s' % self.feats_dir)
        print('max sequence length of caption is', self.cap_max_len)

    def __getitem__(self, ix):

        if self.mode == 'train':
            ix = random.choice(self.splits['train'])
        elif self.mode == 'test':
            ix = self.splits['test'][ix]

        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % ix)))
        fc_feat = np.concatenate(fc_feat, axis=1)

        raw_data = self.captions['video%i' % ix]
        num_cap = len(raw_data)
        cap_mask = np.zeros((num_cap, self.cap_max_len))
        cap_gts = np.zeros((num_cap, self.cap_max_len))
        int_list, eff_list, att_list = [], [], []

        # Load all num_cap gt captions
        for cap_ix in range(num_cap):
            caption = raw_data[cap_ix % len(raw_data)]['final_caption']

            if len(caption) > self.cap_max_len:
                caption = caption[:self.cap_max_len]
                caption[-1] = '<eos>'

            for j, w in enumerate(caption[0: self.cap_max_len]):
                cap_gts[cap_ix, j] = self.cap_word_to_ix.get(w, '1')

            intentions, effects, attributes = raw_data[cap_ix]['intention'], raw_data[cap_ix]['effect'], \
                                              raw_data[cap_ix]['attribute']

            # Concatenate all CMS
            int_str, att_str, eff_str = '', '', ''
            for ints, eff, att in zip(intentions, effects, attributes):
                int_str += ';' + ints[0]
                eff_str += ';' + eff[0]
                att_str += ';' + att[0]

            int_list.append(int_str)
            eff_list.append(eff_str)
            att_list.append(att_str)

        # Insert mask
        cap_mask[(cap_gts != 0)] = 1


        clip_step = 2
        idx_v = list(range(40))
        if self.mode == 'train':
            idx_v_select = []
            for i in idx_v[::clip_step]:
                k = random.randint(0, clip_step - 1)
                if i + k < 40:
                    idx_v_select.append(i + k)
                else:
                    idx_v_select.append(39)
        else:
            idx_v_select = idx_v[::clip_step]
        fc_feat = fc_feat[idx_v_select, :]

        i3d = np.load(
            os.path.join('data/i3d_features', 'msr_vtt-I3D-RGBFeatures-video%i.npy' % ix))  # [5,1024] -> [10,1024]
        l_i3d = list(range(i3d.shape[0]))
        if len(l_i3d) > 10:
            lin_i3d = int(len(l_i3d) / 3)
            lin_i3d2 = len(l_i3d) - 2 * lin_i3d
            if self.mode == 'train':
                i3d_idx = sorted(random.sample(l_i3d[0:lin_i3d], 3)) + sorted(
                    random.sample(l_i3d[lin_i3d:2 * lin_i3d], 3)) + sorted(random.sample(l_i3d[2 * lin_i3d:], 4))
            else:
                i3d_idx = [int(lin_i3d / 3 * 1 - 1), int(lin_i3d / 3 * 2 - 1), int(lin_i3d - 1),
                           int(lin_i3d / 3 * 1 - 1) + lin_i3d, int(lin_i3d / 3 * 2 - 1) + lin_i3d,
                           int(lin_i3d - 1) + lin_i3d,
                           int(lin_i3d2 / 4 * 1 - 1) + lin_i3d * 2, int(lin_i3d2 / 4 * 2 - 1) + lin_i3d * 2,
                           int(lin_i3d2 / 4 * 3 - 1) + lin_i3d * 2, int(lin_i3d2 - 1) + lin_i3d * 2,
                           ]
        else:
            i3d_idx = l_i3d + [l_i3d[-1] for i in range(10 - len(l_i3d))]
        i3d = i3d[i3d_idx, :]

        audio = np.transpose(
            h5py.File(os.path.join('data/audio_features', 'video%i.mp3.soundnet.h5' % ix), 'r')['layer24'],
            (1, 0))
        l_aud = list(range(audio.shape[0]))
        lin_aud = int(len(l_aud) / 3)
        lin_aud2 = len(l_aud) - 2 * lin_aud
        if self.mode == 'train':
            aud_idx = sorted(random.sample(l_aud[0:lin_aud], 3)) + sorted(
                random.sample(l_aud[lin_aud:2 * lin_aud], 3)) + sorted(random.sample(l_aud[2 * lin_aud:], 4))
        else:
            aud_idx = [int(lin_aud / 3 * 1 - 1), int(lin_aud / 3 * 2 - 1), int(lin_aud - 1),
                       int(lin_aud / 3 * 1 - 1) + lin_aud, int(lin_aud / 3 * 2 - 1) + lin_aud,
                       int(lin_aud - 1) + lin_aud,
                       int(lin_aud2 / 4 * 1 - 1) + lin_aud * 2, int(lin_aud2 / 4 * 2 - 1) + lin_aud * 2,
                       int(lin_aud2 / 4 * 3 - 1) + lin_aud * 2, int(lin_aud2 - 1) + lin_aud * 2,
                       ]
        audio = audio[aud_idx, :]

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['cap_labels'] = torch.from_numpy(cap_gts).type(torch.LongTensor)
        data['i3d'] = torch.from_numpy(i3d).type(torch.FloatTensor) # 10,2048
        data['audio'] = torch.from_numpy(audio).type(torch.FloatTensor)  # 10,2048
        data['cap_masks'] = torch.from_numpy(cap_mask).type(torch.FloatTensor)
        data['video_ids'] = 'video%i' % ix

        return data, int_list, eff_list, att_list

    def __len__(self):
        return len(self.splits[self.mode])



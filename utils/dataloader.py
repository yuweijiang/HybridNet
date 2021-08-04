import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py


class VideoDataset(Dataset):
    """

    """

    def tensorize_float(self, obj):
        return torch.from_numpy(obj).type(torch.FloatTensor)

    def tensorize_long(self, obj):
        return torch.from_numpy(obj).type(torch.LongTensor)

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
        super(VideoDataset, self).__init__()
        self.mode = mode

        self.captions = json.load(open(opt['caption_json']))   # V2C_MSR-VTT_caption.json
        cms_info = json.load(open(opt['info_json']))    # v2c_info.json
        self.cms_ix_to_word = cms_info['ix_to_word']
        self.cms_word_to_ix = cms_info['word_to_ix']
        self.splits = cms_info['videos']

        # Load caption dictionary
        cap_info = json.load(open(opt['cap_info_json']))   # msrvtt_new_info.json
        self.cap_ix_to_word = cap_info['ix_to_word']
        self.cap_word_to_ix = cap_info['word_to_ix']

        print('Caption vocab size is ', len(self.cap_ix_to_word))
        print('CMS vocab size is ', len(self.cms_ix_to_word))
        print('number of train videos: ', len(self.splits['train']))
        print('number of test videos: ', len(self.splits['test']))
        print('number of val videos: ', len(self.splits['val']))

        self.feats_dir = opt['feats_dir']   # data/feats/resnet152/
        self.i3d_dir = opt['i3d_dir']
        self.audio_dir = opt['audio_dir']

        print('load feats from %s' % self.feats_dir)

        self.cap_max_len = opt['cap_max_len']  # 28
        self.int_max_len = opt['int_max_len']  # 21
        self.eff_max_len = opt['eff_max_len']  # 26
        self.att_max_len = opt['att_max_len']  # 8

        print('max sequence length of caption is', self.cap_max_len)
        print('max sequence length of intention is', self.int_max_len)
        print('max sequence length of effect is', self.eff_max_len)
        print('max sequence length of attribute is', self.att_max_len)

    def __getitem__(self, ix=False):
        if self.mode == 'train':
            ix = self.splits['train'][ix]

        elif self.mode == 'test':
            ix = self.splits['test'][ix]

        # Load the visual features
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % ix)))
        fc_feat = np.concatenate(fc_feat, axis=1)   # 40,2048

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

        i3d = np.load(os.path.join('data/i3d_features', 'msr_vtt-I3D-RGBFeatures-video%i.npy' % ix))   # [5,1024] -> [10,1024]
        l_i3d = list(range(i3d.shape[0]))
        if len(l_i3d) > 10:
            lin_i3d = int(len(l_i3d) / 3)
            lin_i3d2 = len(l_i3d) - 2 * lin_i3d
            if self.mode == 'train':
                i3d_idx = sorted(random.sample(l_i3d[0:lin_i3d], 3)) + sorted(random.sample(l_i3d[lin_i3d:2*lin_i3d], 3)) + sorted(random.sample(l_i3d[2*lin_i3d:], 4))
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

        audio = np.transpose(h5py.File(os.path.join('data/audio_features', 'video%i.mp3.soundnet.h5' % ix), 'r')['layer24'],(1, 0))  # [4, 1024]  -> [10,1024]
        l_aud = list(range(audio.shape[0]))
        lin_aud = int(len(l_aud) / 3)
        lin_aud2 = len(l_aud) - 2 * lin_aud
        if self.mode == 'train':
            aud_idx = sorted(random.sample(l_aud[0:lin_aud], 3)) + sorted(random.sample(l_aud[lin_aud:2 * lin_aud], 3)) + sorted(random.sample(l_aud[2 * lin_aud:], 4))
        else:
            aud_idx= [int(lin_aud / 3 * 1 - 1), int(lin_aud / 3 * 2 - 1), int(lin_aud - 1),
             int(lin_aud / 3 * 1 - 1) + lin_aud, int(lin_aud / 3 * 2 - 1) + lin_aud, int(lin_aud - 1) + lin_aud,
             int(lin_aud2 / 4 * 1 - 1) + lin_aud * 2, int(lin_aud2 / 4 * 2 - 1) + lin_aud * 2,
             int(lin_aud2 / 4 * 3 - 1) + lin_aud * 2, int(lin_aud2 - 1) + lin_aud * 2,
             ]
        audio = audio[aud_idx, :]

        # Placeholder for returning parameters
        cap_mask = np.zeros(self.cap_max_len)  # 28
        int_mask = np.zeros(self.int_max_len)  # 21
        eff_mask = np.zeros(self.eff_max_len)  # 26
        att_mask = np.zeros(self.att_max_len)  # 8

        cap_gts = np.zeros(self.cap_max_len)
        int_gts = np.zeros(self.int_max_len)
        eff_gts = np.zeros(self.eff_max_len)
        att_gts = np.zeros(self.att_max_len)

        if 'video%i' % ix not in self.captions:
            print(ix in self.splits['train'])

        assert 'video%i' % ix in self.captions
        raw_data = self.captions['video%i' % ix]

        # Random pick out one caption in Training mode
        cap_ix = random.randint(0, len(raw_data) - 1)

        # Pop out Cap, Int, Eff and Att
        caption = self.captions['video%i' % ix][cap_ix]['final_caption']

        intentions = self.captions['video%i' % ix][cap_ix]['intention']
        intention = intentions[random.randint(0, len(intentions)-1)][1]

        effects = self.captions['video%i' % ix][cap_ix]['effect']
        effect = effects[random.randint(0, len(effects)-1)][1]

        attributes = self.captions['video%i' % ix][cap_ix]['attribute']
        attribute = attributes[random.randint(0, len(attributes)-1)][1]

        # Trunk the tokens if it exceed the maximum limitation
        if len(caption) > self.cap_max_len:
            caption = caption[:self.cap_max_len]
            caption[-1] = '<eos>'

        if len(intention) > self.int_max_len:
            intention = intention[:self.int_max_len]
            intention[-1] = '<eos>'

        if len(effect) > self.eff_max_len:
            effect = effect[:self.eff_max_len]
            effect[-1] = '<eos>'

        if len(attribute) > self.att_max_len:
            attribute = attribute[:self.att_max_len]
            attribute[-1] = '<eos>'

        # Tokenize it
        for j, w in enumerate(caption):
            cap_gts[j] = self.cap_word_to_ix.get(w, '1')

        for j, w in enumerate(intention):
            int_gts[j] = self.cms_word_to_ix.get(w, '1')

        for j, w in enumerate(effect):
            eff_gts[j] = self.cms_word_to_ix.get(w, '1')

        for j, w in enumerate(attribute):
            att_gts[j] = self.cms_word_to_ix.get(w, '1')

        # Mask out additional positions
        non_zero = (cap_gts == 0).nonzero()
        if len(non_zero[0]) != 0: cap_mask[:int(non_zero[0][0])] = 1
        else: cap_mask += 1

        non_zero = (int_gts == 0).nonzero()
        if len(non_zero[0]) != 0: int_mask[:int(non_zero[0][0])] = 1
        else: int_mask += 1

        non_zero = (eff_gts == 0).nonzero()
        if len(non_zero[0]) != 0: eff_mask[:int(non_zero[0][0])] = 1
        else: eff_mask += 1

        non_zero = (att_gts == 0).nonzero()
        if len(non_zero[0]) != 0: att_mask[:int(non_zero[0][0])] = 1
        else: att_mask += 1

        # Convert to Tensors
        data = {}
        data['fc_feats'] = self.tensorize_float(fc_feat)   # 20,2048
        data['i3d'] = self.tensorize_float(i3d)  # 10,2048
        data['audio'] = self.tensorize_float(audio)  # 10,2048
        data['cap_labels'] = self.tensorize_long(cap_gts)  # 28
        data['cap_masks'] = self.tensorize_float(cap_mask) # [1,1,1,1,1,0...]
        data['int_labels'] = self.tensorize_long(int_gts)  # 21
        data['int_masks'] = self.tensorize_float(int_mask)
        data['eff_labels'] = self.tensorize_long(eff_gts)  # 26
        data['eff_masks'] = self.tensorize_float(eff_mask)
        data['att_labels'] = self.tensorize_long(att_gts)  # 8
        data['att_masks'] = self.tensorize_float(att_mask)
        data['video_ids'] = 'video%i' % ix

        return data

    def __len__(self):
        return len(self.splits[self.mode])
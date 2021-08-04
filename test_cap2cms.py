import json
import torch
import random
import numpy as np
from opts import *
from model.Model import HybirdNet as Model
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Constants import *
from model.transformer.Translator import translate_batch_cap, translate_batch_cap2cms
import nltk
from utils.utils import *

import sys
sys.path.append("utils/pycocoevalcap/")


def pos_emb_generation(visual_feats):
    '''
        Generate the position embedding input for Transformers.
    '''
    seq = list(range(1, visual_feats.shape[1] + 1))
    src_pos = torch.tensor([seq] * visual_feats.shape[0]).cuda()
    return src_pos


def list_to_sentence(list):
    sentence = ''
    for element in list:
        sentence += ' ' + element
    return sentence


def test(loader, model, opt, cap_vocab, cms_vocab):

    gts = {}
    res = {}

    json_dict = {}
    total_cms = set()
    ppl_scores = []

    eval_id = 0
    for batch_id, data in enumerate(loader):

        fc_feats = data['fc_feats'].cuda()
        cap_labels = data['cap_labels'].cuda()
        video_ids = data['video_ids']

        i3d = data['i3d'].cuda()  # 10*1024
        audio = data['audio'].cuda()  # 10*1024

        with torch.no_grad():
            # Beam Search Starts From Here
            try:
                batch_hyp = translate_batch_cap(model, fc_feats, i3d, audio, opt)
            except:
                continue

        cap_res = torch.zeros_like(cap_labels)
        for idx in range(len(batch_hyp)):
            cap_res[idx][0] = CAP_BOS

            if len(batch_hyp[idx][0]) < opt['cap_max_len']:
                for word_idx in range(0, len(batch_hyp[idx][0])):
                    cap_res[idx][word_idx+1] = batch_hyp[idx][0][word_idx]
            else:
                for word_idx in range(0, len(batch_hyp[idx][0])-1):
                    cap_res[idx][word_idx+1] = batch_hyp[idx][0][word_idx]

        with torch.no_grad():
            cms_batch_hyp = translate_batch_cap2cms(model, fc_feats, cap_res, i3d, audio, opt)


        # Stack all GTs captions
        references = []  # 30
        for video in video_ids:
            video_caps = []
            for cap in opt['captions'][video]:
                if opt['cms'] == 'int':
                    for _ in cap['intention']:
                        video_caps.append(list_to_sentence(cap['final_caption'][1:-1] + _[1][1:-1]))
                if opt['cms'] == 'eff':
                    for _ in cap['effect']:
                        video_caps.append(list_to_sentence(cap['final_caption'][1:-1] + _[1][1:-1]))
                if opt['cms'] == 'att':
                    for _ in cap['attribute']:
                        video_caps.append(list_to_sentence(cap['final_caption'][1:-1] + _[1][1:-1]))
            references.append(video_caps)
            # Stack all Predicted Captions

        hypotheses = []
        for predict, predict_cms in zip(batch_hyp, cms_batch_hyp):
            _ = []
            if CAP_EOS in predict[0]:
                sep_id = predict[0].index(CAP_EOS)
            else:
                sep_id = -1
            for word in predict[0][0: sep_id]:
                _.append(cap_vocab[str(word)])

            if CAP_EOS in predict_cms[0]:
                sep_id = predict_cms[0].index(CAP_EOS)
            else:
                sep_id = -1
            for word in predict_cms[0][0: sep_id]:
                _.append(cms_vocab[str(word)])
            hypotheses.append(list_to_sentence(_))


        for random_id in range(1):
            print('Generated Caption:', hypotheses[random_id])
            print('GT:', references[random_id][0])
            print('\n')
            print(batch_id, ' ', batch_id * opt['batch_size'], ' out of ', '3010')

        for random_id in range(cap_labels.shape[0]):
            res[eval_id] = [hypotheses[random_id]]
            gts[eval_id] = references[random_id]
            eval_id += 1

            json_dict[video_ids[random_id] + '_' + str(eval_id)] = {'gt': references[random_id], 'pred': hypotheses[random_id]}
            # print(references[random_id], 'pred', hypotheses[random_id])

            ppl_corpus = ''
            for c in references[random_id]:
                total_cms.add(c.lower())
                ppl_corpus += ' ' + c.lower()
            tokens = nltk.word_tokenize(ppl_corpus)
            unigram_model = unigram(tokens)
            ppl_scores.append(perplexity(hypotheses[random_id].lower(), unigram_model))


    # Compute PPL score
    print('Perplexity score: ', sum(ppl_scores) / len(ppl_scores))

    avg_bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
    avg_cider_score, cider_scores = Cider().compute_score(gts, res)
    avg_meteor_score, meteor_scores = Meteor().compute_score(gts, res)
    avg_rouge_score, rouge_scores = Rouge().compute_score(gts, res)
    print('C, M, R, B:', avg_cider_score, avg_meteor_score, avg_rouge_score, avg_bleu_score)



def main(opt):
    dataset = VideoDataset(opt, 'test')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False)
    opt['cms_vocab_size'] = dataset.get_cms_vocab_size()
    opt['cap_vocab_size'] = dataset.get_cap_vocab_size()

    if opt['cms'] == 'int':
        cms_text_length = opt['int_max_len']
    elif opt['cms'] == 'eff':
        cms_text_length = opt['eff_max_len']
    else:
        cms_text_length = opt['att_max_len']

    cms_int_text_length = opt['int_max_len']
    cms_eff_text_length = opt['eff_max_len']
    cms_att_text_length = opt['att_max_len']

    model = Model(
        dataset.get_cap_vocab_size(),
        dataset.get_cms_vocab_size(),
        cap_max_seq=opt['cap_max_len'],
        cms_max_seq_int=cms_int_text_length,
        cms_max_seq_eff=cms_eff_text_length,
        cms_max_seq_att=cms_att_text_length,
        tgt_emb_prj_weight_sharing=False,
        vis_emb=opt['dim_vis_feat'],
        rnn_layers=opt['rnn_layer'],
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=opt['dropout'])

    if len(opt['load_checkpoint']) != 0:
        state_dict = torch.load(opt['load_checkpoint'])
        model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()
    test(dataloader, model, opt, dataset.get_cap_vocab(), dataset.get_cms_vocab())


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    opt['captions'] = json.load(open(opt['caption_json']))
    opt['batch_size'] = 30
    main(opt)
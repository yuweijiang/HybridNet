''' Training Scropt for V2C captioning task. '''

import os
import numpy as np
from opts import *
from utils.utils import *
import torch.optim as optim
from model.Model import HybirdNet as Model
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Optim import ScheduledOptim
import torch

def train(loader, model, optimizer, opt, cap_vocab, cms_vocab):

    model.train()

    for epoch in range(1, opt['epochs']+1):
        iteration = 0
        cap_n_correct_total = 0
        cms_int_n_correct_total = 0
        cms_eff_n_correct_total = 0
        cms_att_n_correct_total = 0
        n_word_total = 0
        cms_int_n_word_total = 0
        cms_eff_n_word_total = 0
        cms_att_n_word_total = 0
        cap_train_loss_total = 0
        cms_int_train_loss_total = 0
        cms_eff_train_loss_total = 0
        cms_att_train_loss_total = 0

        for data in loader:
            torch.cuda.synchronize()

            cms_labels_int = data['int_labels']
            cms_labels_eff = data['eff_labels']
            cms_labels_att = data['att_labels']

            if opt['cuda']:
                fc_feats = data['fc_feats'].cuda()
                i3d = data['i3d'].cuda()
                audio = data['audio'].cuda()
                cap_labels = data['cap_labels'].cuda()
                cms_labels_int = cms_labels_int.cuda()
                cms_labels_eff = cms_labels_eff.cuda()
                cms_labels_att = cms_labels_att.cuda()

            optimizer.zero_grad()
            cap_pos = pos_emb_generation(cap_labels)
            cms_pos_int = pos_emb_generation(cms_labels_int)
            cms_pos_eff = pos_emb_generation(cms_labels_eff)
            cms_pos_att = pos_emb_generation(cms_labels_att)


            cap_probs, cms_int_probs, cms_eff_probs, cms_att_probs = model(fc_feats, i3d, audio, cap_labels, cap_pos, cms_labels_int, cms_pos_int,
                                         cms_labels_eff, cms_pos_eff, cms_labels_att, cms_pos_att)

            # note: currently we just used most naive cross-entropy as training objective,
            # advanced loss func. like SELF-CRIT, different loss weights or stronger video feature
            # may lead performance boost, however is not the goal of this work.
            cap_loss, cap_n_correct = cal_performance(cap_probs, cap_labels[:, 1:], smoothing=True)
            cms_int_loss, cms_int_n_correct = cal_performance(cms_int_probs, cms_labels_int[:, 1:], smoothing=True)
            cms_eff_loss, cms_eff_n_correct = cal_performance(cms_eff_probs, cms_labels_eff[:, 1:], smoothing=True)
            cms_att_loss, cms_att_n_correct = cal_performance(cms_att_probs, cms_labels_att[:, 1:], smoothing=True)

            # compute the token prediction Acc.
            non_pad_mask = cap_labels[:, 1:].ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()

            cms_int_non_pad_mask = cms_labels_int[:, 1:].ne(Constants.PAD)
            cms_int_n_word = cms_int_non_pad_mask.sum().item()

            cms_eff_non_pad_mask = cms_labels_eff[:, 1:].ne(Constants.PAD)
            cms_eff_n_word = cms_eff_non_pad_mask.sum().item()

            cms_att_non_pad_mask = cms_labels_att[:, 1:].ne(Constants.PAD)
            cms_att_n_word = cms_att_non_pad_mask.sum().item()

            cap_loss /= n_word
            cms_int_loss /= cms_int_n_word
            cms_eff_loss /= cms_eff_n_word
            cms_att_loss /= cms_att_n_word

            loss = cap_loss + cms_int_loss + cms_eff_loss + cms_att_loss

            loss.backward()
            optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

            # update parameters
            cap_train_loss = cap_loss.item()
            cms_int_train_loss = cms_int_loss.item()
            cms_eff_train_loss = cms_eff_loss.item()
            cms_att_train_loss = cms_att_loss.item()

            # multi-gpu case, not necessary in newer PyTorch version or on single GPU.
            if opt['cuda']: torch.cuda.synchronize()

            iteration += 1
            cap_n_correct_total += cap_n_correct
            cms_int_n_correct_total += cms_int_n_correct
            cms_eff_n_correct_total += cms_eff_n_correct
            cms_att_n_correct_total += cms_att_n_correct
            n_word_total += n_word
            cms_int_n_word_total += cms_int_n_word
            cms_eff_n_word_total += cms_eff_n_word
            cms_att_n_word_total += cms_att_n_word
            cap_train_loss_total += cap_train_loss
            cms_int_train_loss_total += cms_int_train_loss
            cms_eff_train_loss_total += cms_eff_train_loss
            cms_att_train_loss_total += cms_att_train_loss

            if iteration % opt['print_loss_every'] ==0:
                print('iter %d (epoch %d), cap_train_loss = %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
                      ' current step = %d, current lr = %.3E, cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f'
                      % (iteration, epoch, cap_train_loss, cms_int_train_loss, cms_eff_train_loss, cms_att_train_loss, optimizer.n_current_steps,
                         optimizer._optimizer.param_groups[0]['lr'],
                         cap_n_correct/n_word, cms_int_n_correct/cms_int_n_word, cms_eff_n_correct/cms_eff_n_word, cms_att_n_correct/cms_att_n_word))

                # show the intermediate generations
                if opt['show_predict']:
                    cap_pr, cap_gt = show_prediction(cap_probs, cap_labels[:, :-1], cap_vocab, caption=True)
                    cms_int_pr, cms_int_gt = show_prediction(cms_int_probs, cms_labels_int[:, :-1], cms_vocab,
                                                             caption=False)
                    cms_eff_pr, cms_eff_gt = show_prediction(cms_eff_probs, cms_labels_eff[:, :-1], cms_vocab,
                                                             caption=False)
                    cms_att_pr, cms_att_gt = show_prediction(cms_att_probs, cms_labels_att[:, :-1], cms_vocab,
                                                             caption=False)
                    print(' \n')

                with open(opt['info_path'], 'a') as f:
                    f.write('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,\n'
                            % (epoch, cap_train_loss / iteration, cms_int_train_loss / iteration,
                               cms_eff_train_loss / iteration,
                               cms_att_train_loss / iteration))
                    f.write('\n %s \n %s' % (cap_pr, cap_gt))
                    f.write('\n %s \n %s' % (cms_int_pr, cms_int_gt))
                    f.write('\n %s \n %s' % (cms_eff_pr, cms_eff_gt))
                    f.write('\n %s \n %s' % (cms_att_pr, cms_att_gt))
                    f.write('\n')

        print('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
              'cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f\n'
            % (epoch, cap_train_loss_total / iteration, cms_int_train_loss_total / iteration, cms_eff_train_loss_total / iteration, cms_att_train_loss_total / iteration,
               cap_n_correct_total/n_word_total, cms_int_n_correct_total/cms_int_n_word_total, cms_eff_n_correct_total/cms_eff_n_word_total, cms_att_n_correct_total/cms_att_n_word_total))


        if epoch % opt['save_checkpoint_every'] == 0:

            # save the checkpoint
            model_path = os.path.join(opt['output_dir'],
                                      '{}_{}.pth'
                                      .format(opt['output_dir'].split('/')[-1], epoch))

            if torch.__version__ == '1.3.1':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

            print('model saved to %s' % model_path)
            with open(opt['model_info_path'], 'a') as f:
                f.write('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
                        'cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f\n'
                        % (epoch, cap_train_loss_total / iteration, cms_int_train_loss_total / iteration, cms_eff_train_loss_total / iteration, cms_att_train_loss_total / iteration,
                           cap_n_correct_total/n_word_total, cms_int_n_correct_total/cms_int_n_word_total, cms_eff_n_correct_total/cms_eff_n_word_total, cms_att_n_correct_total/cms_att_n_word_total))


def main(opt):
    # load and define dataloader
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)

    opt['cms_vocab_size'] = dataset.get_cms_vocab_size()
    opt['cap_vocab_size'] = dataset.get_cap_vocab_size()


    cms_int_text_length = opt['int_max_len']
    cms_eff_text_length = opt['eff_max_len']
    cms_att_text_length = opt['att_max_len']

    # model initialization.
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

    # number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of learnable parameters are {}'.format(params))

    if opt['cuda']: model = model.cuda()

    # resume from previous checkpoint if indicated
    if opt['load_checkpoint'] and opt['resume']:
        cap_state_dict = torch.load(opt['load_checkpoint'])
        model_dict = model.state_dict()
        model_dict.update(cap_state_dict)
        model.load_state_dict(model_dict)

    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    # note: though we set the init learning rate as np.power(d_model, -0.5),
    # grid search indicates different LR may improve the results.
    opt['init_lr'] = round(optimizer.init_lr, 3)

    # create checkpoint output directory
    dir = opt['output_dir']
    if not os.path.exists(dir): os.makedirs(dir)

    # save the model snapshot to local
    info_path = os.path.join(dir, 'iteration_info_log.log')
    print('model architecture saved to {} \n {}'.format(info_path, str(model)))
    with open(info_path, 'a') as f:
        f.write(str(model))
        f.write('\n')
        f.write(str(params))
        f.write('\n')

    # log file directory
    opt['info_path'] = info_path
    opt['model_info_path'] = os.path.join(opt['output_dir'],
                                          'checkpoint_loss_log.log')

    train(dataloader, model, optimizer, opt, dataset.get_cap_vocab(), dataset.get_cms_vocab())

if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    main(opt)
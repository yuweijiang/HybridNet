''' Define the Transformer model '''
import torch
import numpy as np
import torch.nn as nn
import model.transformer.Constants as Constants
from model.transformer.Layers import EncoderLayer, DecoderLayer


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1).cuda()


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask.cuda()


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask.cuda()


class Encoder_video(nn.Module):
    '''
    A encoder model with self attention mechanism.
    '''
    def __init__(self, d_vison=2048,  d_i3d=1024, d_audio=1024, d_model=512):

        super().__init__()

        self.vih = nn.Linear(d_vison, d_model)
        self.d2h = nn.Linear(d_i3d, d_model)
        self.a2h = nn.Linear(d_audio, d_model)
        self.vih_dropout = nn.Dropout(0.1)
        self.d2h_dropout = nn.Dropout(0.1)
        self.a2h_dropout = nn.Dropout(0.1)
        self.position_enc = nn.Embedding(3, d_model)
        self.rnn_vis = nn.GRU(d_model, d_model, 1, batch_first=True, dropout=0.3)
        self.rnn_i3d = nn.GRU(d_model, d_model, 1, batch_first=True, dropout=0.3)
        self.rnn_aud = nn.GRU(d_model, d_model, 1, batch_first=True, dropout=0.3)


    def forward(self, src_emb, i3d_emb, aud_emb):

        src_emb = self.vih_dropout(self.vih(src_emb))
        i3d_emb = self.d2h_dropout(self.d2h(i3d_emb))
        aud_emb = self.a2h_dropout(self.a2h(aud_emb))

        # -- Forward
        self.rnn_vis.flatten_parameters()
        self.rnn_i3d.flatten_parameters()
        self.rnn_aud.flatten_parameters()
        state1 = None
        state2 = None
        state3 = None
        src_emb, *_ = self.rnn_vis(src_emb, state1)
        i3d_emb, *_ = self.rnn_i3d(i3d_emb, state2)
        aud_emb, *_ = self.rnn_aud(aud_emb, state3)
        src_emb = src_emb + self.position_enc(torch.full((src_emb.shape[0], 20), 0).long().cuda())
        i3d_emb = i3d_emb + self.position_enc(torch.full((src_emb.shape[0], 10), 1).long().cuda())
        aud_emb = aud_emb + self.position_enc(torch.full((src_emb.shape[0], 10), 2).long().cuda())

        enc_output = torch.cat((src_emb, i3d_emb, aud_emb), dim=1)

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        src_tmp = torch.ones(src_seq.shape[0], src_seq.shape[1]).cuda()
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_tmp, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, vis_emb=2048,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        self.vis_emb = nn.Linear(vis_emb, d_model)

        self.encoder = Encoder(
            len_max_seq=40,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, ' \
            'the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_emb, src_pos, tgt_seq, tgt_pos):

        src_emb = self.vis_emb(src_emb)
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_emb, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_emb, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


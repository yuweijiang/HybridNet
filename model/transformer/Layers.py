''' Define the Layers '''
import torch.nn as nn
from model.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward,  MultiHeadAttention_rmatt_ev_cnn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn



class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention_rmatt_ev_cnn(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention_rmatt_ev_cnn(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None, memory=None, self_attn_pre=None, dec_enc_attn_pre=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask, memory=memory, attn_pre=self_attn_pre, fg=1)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask, memory=memory, attn_pre=dec_enc_attn_pre, fg=2)
        dec_output *= non_pad_mask    # [b,len_q,d_model]  [n,len_q,1]

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
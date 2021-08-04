import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))   # [b*h, len_q, len_k]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)  # [b*h, len_q, len_k]
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [b*h, len_q, dim]

        return output, attn

class ScaledDotProductAttention_rmatt(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, q_rm=None):

        attn = torch.bmm(q, k.transpose(1, 2))   # [b*h, len_q, len_k]
        attn = attn / self.temperature

        attn_rm = torch.bmm(q_rm, k.transpose(1, 2))
        attn_rm = attn_rm / self.temperature

        attn = attn * 0.5 + attn_rm * 0.5

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)  # [b*h, len_q, len_k]
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [b*h, len_q, dim]

        return output, attn


class ScaledDotProductAttention_rmatt_ev_cnn(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''

    def __init__(self, temperature, n_head=8, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.n_head = n_head
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.att_conv = nn.Conv2d(in_channels=n_head, out_channels=n_head,
                                  kernel_size=3, padding=1, bias=False)


    def sift_down_right(self, src, filled):
        src[..., 1:, 1:] = src[..., :-1, :-1].clone()
        src[..., 0, ::] = filled[..., 0, ::]
        src[..., ::, 0] = filled[..., ::, 0]
        return src

    def sift_down(self, src, filled):
        src[..., 1:, ::] = src[..., :-1, ::].clone()
        src[..., 0, ::] = filled[..., 0, ::]
        return src

    def forward(self, q, k, v, mask=None, q_rm=None, attn_pre=None, fg=1):

        attn = torch.bmm(q, k.transpose(1, 2))   # [b*h, len_q, len_k]
        attn = attn / self.temperature

        attn_rm = torch.bmm(q_rm, k.transpose(1, 2))
        attn_rm = attn_rm / self.temperature

        a = 0.1
        b = 0.4
        c = 0.1

        # nocnn
        # attn = attn_rm * b + attn * (1 - b)

        # cnn
        if fg == 1:
            attn_cnn = attn_rm.masked_fill(mask, 0)
            attn_cnn = attn_cnn.view(-1, self.n_head, attn_rm.size(-2), attn_rm.size(-1))
            attn_cnn = nn.functional.relu(self.att_conv(attn_cnn))
            attn_cnn = attn_cnn.view(-1, attn_rm.size(-2), attn_rm.size(-1))
            attn_cnn = self.sift_down_right(attn_cnn, attn_rm)
        elif fg == 2:
            attn_cnn = attn_rm.masked_fill(mask, 0)
            attn_cnn = attn_cnn.view(-1, self.n_head, attn_rm.size(-2), attn_rm.size(-1))
            attn_cnn = nn.functional.relu(self.att_conv(attn_cnn))
            attn_cnn = attn_cnn.view(-1, attn_rm.size(-2), attn_rm.size(-1))
            attn_cnn = self.sift_down(attn_cnn, attn_rm)

        attn_cnn = attn_cnn * c + attn_rm * (1-c)
        attn = attn_cnn * b + attn * (1-b)

        if attn_pre is not None:
            attn = attn_pre * a + attn * (1-a)
        else:
            attn = attn

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)  # [b*h, len_q, len_k]
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [b*h, len_q, dim]

        return output, attn


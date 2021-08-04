import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, n_head, attn_dropout=0.1):
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

    def forward(self, q, k, v, attn_pre=None, mask=None, fg=0):
        # fg：0无    1selfenc   2encdec
        # def forward(self, q, k, v, mask=None):
        #  (n*b) x lq x dq    (n*b) x lk x dk    (n*b) x lk x dk

        attn = torch.bmm(q, k.transpose(1, 2))   # [b*h, len_q, len_k]

        attn = attn / self.temperature

        # if mask is not None:
        #     attn = attn.masked_fill(mask, -np.inf)
        #
        # attn = self.softmax(attn)  # [b*h, len_q, len_k]
        attn_now = attn

        #  ev
        a = 0.5    # 保留上一阶段的权重 0.5
        b = 0.1    # 保留cnn后的权重 0.1
        if attn_pre is not None:
            attn = attn_pre * a + attn_now * (1-a)
        else:
            attn = attn_now

        if fg == 0:
            attn_cnn = attn.masked_fill(mask, 0)
            attn_cnn = attn_cnn.view(-1, self.n_head, attn.size(-2), attn.size(-1))
            attn_cnn = nn.functional.relu(self.att_conv(attn_cnn))
            attn_cnn = attn_cnn.view(-1, attn.size(-2), attn.size(-1))

        elif fg == 1:
            attn_cnn = attn.masked_fill(mask, 0)
            attn_cnn = attn_cnn.view(-1, self.n_head, attn.size(-2), attn.size(-1))
            attn_cnn = nn.functional.relu(self.att_conv(attn_cnn))
            attn_cnn = attn_cnn.view(-1, attn.size(-2), attn.size(-1))
            attn_cnn = self.sift_down_right(attn_cnn, attn)
        elif fg == 2:
            attn_cnn = attn.masked_fill(mask, 0)
            attn_cnn = attn_cnn.view(-1, self.n_head, attn.size(-2), attn.size(-1))
            attn_cnn = nn.functional.relu(self.att_conv(attn_cnn))
            attn_cnn = attn_cnn.view(-1, attn.size(-2), attn.size(-1))
            attn_cnn = self.sift_down(attn_cnn, attn)


        attn = attn_cnn * b + attn * (1-b)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [b*h, len_q, dim]

        return output, attn


class ScaledDotProductAttention2(nn.Module):
    '''
        Scaled Dot-Product Attention
        our卷积
          # fg：0无    1self-enc   2enc-dec
    '''

    def __init__(self, temperature, n_head, attn_dropout=0.1, fg=1):

        super().__init__()
        self.temperature = temperature
        self.n_head = n_head
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.fg = fg

        if fg == 1:
            self.att_linear = nn.Linear(27, 27)
        else:
            self.att_linear = nn.Linear(40, 40)

        self.att_conv = nn.Conv2d(in_channels=n_head, out_channels=n_head,
                                  kernel_size=(27,1), padding=(26,0), bias=False)


    def forward(self, q, k, v, attn_pre=None, mask=None, fg=0):
        # fg：0无    1self-enc   2enc-dec
        # def forward(self, q, k, v, mask=None):
        #  (n*b) x lq x dq    (n*b) x lk x dk    (n*b) x lk x dk

        # print(q.shape, k.shape)

        attn = torch.bmm(q, k.transpose(1, 2))   # [b*h, len_q, len_k]

        attn = attn / self.temperature

        # if mask is not None:
        #     attn = attn.masked_fill(mask, -np.inf)
        #
        # attn = self.softmax(attn)  # [b*h, len_q, len_k]
        attn_now = attn

        #  ev
        a = 0.5    # 保留上一阶段的权重 0.5
        b = 0.1    # 保留cnn后的权重 0.1
        if attn_pre is not None:
            attn = attn_pre * a + attn_now * (1-a)
        else:
            attn = attn_now

        attn_cnn = attn.masked_fill(mask, 0)  # [b*n, q, k]
        attn_cnn = self.att_linear(attn_cnn)  # [b*n, q, k]
        attn_cnn = attn_cnn.view(-1, self.n_head, attn.size(-2), attn.size(-1))  # [b, n, q, k]
        attn_cnn = nn.functional.relu(self.att_conv(attn_cnn)[:, :, :attn_cnn.size(-2), :])
        attn_cnn = attn_cnn.view(-1, attn.size(-2), attn.size(-1))


        attn = attn_cnn * b + attn * (1-b)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [b*h, len_q, dim]

        return output, attn

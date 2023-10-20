import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from layers.basic_att import BasicAtt
import ipdb

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        h_k = 8
        h = 8
        h_v = 8
        self.attention_last = nn.Linear(mid_dims[-2], 1)                  
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])
        self.proj_l = nn.Linear(h_k, h)                                    # ! Linear projection P_l (h_k, h),
        self.proj_w = nn.Linear(h, h_v)                                    # ! Linear projection P_w (h, h_v)

    def forward(self, att_map, att_mask, value1, value2):
        print(att_map.shape)
        print(self.attention_basic)
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)                  # * att_map: batch_size * h_k * 196 * 64 ;  alpha_spatial: batch_size * h_k * 196 * 1
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)
        
        # print('alpha_spatial_1: ', alpha_spatial.shape)
        alpha_spatial = alpha_spatial.squeeze(-1)                  
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        # print('att_mask: ', att_mask.shape)
        alpha_spatial = alpha_spatial.permute(0,2,1)                # TODO
        # print(alpha_spatial.shape)
        alpha_spatial = self.proj_l(alpha_spatial)
        alpha_spatial = self.proj_w(alpha_spatial)
        alpha_spatial = alpha_spatial.permute(0,2,1)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)            # TODO    dim=-1
        # print('alpha_spatial_1: ', alpha_spatial)
        # alpha_spatial = self.proj_w(alpha_spatial)
        # print('alpha_spatial_2: ', alpha_spatial)
        # alpha_spatial = alpha_spatial.permute(0,2,1)
        # print('alpha_spatial_2: ', alpha_spatial)


        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn

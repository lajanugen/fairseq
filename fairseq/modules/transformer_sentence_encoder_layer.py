from pdb import set_trace as bp
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)

def layer_norm(x, weight, bias, eps):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True, unbiased=False)
    return weight * ((x - mean) / (std + eps)) + bias

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
        task_emb_cond_type: str = 'cls_token'
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.task_emb_cond_type = task_emb_cond_type 

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        task_emb: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.task_emb_cond_type == 'adapt':
            split_size = task_emb.shape[-1] // 2
            adapter_weights_1 = task_emb.split(split_size, -1)[0]
            x = self.adapter(x, adapter_weights_1)
        x = residual + x

        if self.task_emb_cond_type == 'norm':
            split_shp = task_emb.shape[-1] // 4
            task_emb_split = task_emb.split(split_shp, -1)
            layer_norm_weight_1, layer_norm_bias_1 = task_emb_split[:2]
            layer_norm_weight_2, layer_norm_bias_2 = task_emb_split[2:]

        if self.task_emb_cond_type == 'norm':
            x = layer_norm(x, layer_norm_weight_1, layer_norm_bias_1, eps=1e-5)
        else:
            x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.task_emb_cond_type == 'adapt':
            split_size = task_emb.shape[-1] // 2
            adapter_weights_2 = task_emb.split(split_size, -1)[1]
            x = self.adapter(x, adapter_weights_2)
        x = residual + x
        if self.task_emb_cond_type == 'norm':
            x = layer_norm(x, layer_norm_weight_2, layer_norm_bias_2, eps=1e-5)
        else:
            x = self.final_layer_norm(x)
        return x, attn

    def adapter(self, x, adapter_weights):

        in_size = self.embedding_dim
        # out_size = self.adapter_size
        out_size = (adapter_weights.size(-1) - in_size) // (2 * in_size + 1)
        assert 2 * in_size * out_size + in_size + out_size == adapter_weights.size(-1)

        down_weights, down_bias, up_weights, up_bias = adapter_weights.split(
            [in_size * out_size, out_size, in_size * out_size, in_size], -1)

        down_weights = down_weights.view(-1, in_size, out_size)
        up_weights = up_weights.view(-1, out_size, in_size)

        # down_proj = x.transpose(0, 1).bmm(down_weights) + down_bias.transpose(0, 1)
        if down_weights.shape[0] == 1:
            down_proj = x.transpose(0, 1).matmul(down_weights) + down_bias.unsqueeze(1)
        else:
            down_proj = x.transpose(0, 1).bmm(down_weights) + down_bias.unsqueeze(1)
        # B x T x d

        down_proj = self.activation_fn(down_proj)

        # up_proj = down_proj.bmm(up_weights) + up_bias.transpose(0, 1)
        if up_weights.shape[0] == 1:
            up_proj = down_proj.matmul(up_weights) + up_bias.unsqueeze(1)
        else:
            up_proj = down_proj.bmm(up_weights) + up_bias.unsqueeze(1)
        # B x T x D

        return up_proj.transpose(0, 1) + x


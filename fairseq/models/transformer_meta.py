from pdb import set_trace as bp
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class TransformerDecoderMeta(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        self.input_embed_dim = input_embed_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim
        self.dictionary = dictionary
        self.encoder_embed_dim = args.encoder_embed_dim
        self.freeze_bottom_layers = args.freeze_bottom_layers
        self.task_emb_layer = args.task_emb_layer
        self.split_task_emb = getattr(args, 'split_task_emb', None)
        if self.split_task_emb:
            self.encoder_embed_dim *= args.decoder_layers

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn or (layer_idx <= self.task_emb_layer))
            for layer_idx in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.training_mode = args.training_mode
        self.task_emb_cond_type = args.task_emb_cond_type
        self.LMinit = args.LMinit

        if args.training_mode == 'multitask':
            self.task_embeddings = nn.Embedding(args.max_tasks, self.encoder_embed_dim)
            self.task_embeddings_eval = nn.Embedding(args.max_tasks, self.encoder_embed_dim)

        elif 'meta' in args.training_mode:
            self.task_embeddings = nn.Embedding(args.max_tasks, self.encoder_embed_dim)
            self.task_embeddings_eval = nn.Embedding(args.max_tasks, self.encoder_embed_dim)
            self.z_optimizer = optim.Adam(self.task_embeddings.parameters(), lr=args.z_lr)
            self.z_optimizer_eval = optim.Adam(self.task_embeddings_eval.parameters(), lr=args.z_lr)

        elif args.training_mode == 'single_task':
            self.task_embedding_init = nn.Parameter(torch.zeros(self.encoder_embed_dim))

        self.task_emb_proj = None
        if args.training_mode != 'task_agnostic' and args.task_emb_cond_type == 'decoder':
            if self.encoder_embed_dim != args.decoder_embed_dim: 
                self.task_emb_proj = nn.Linear(self.encoder_embed_dim, args.decoder_embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, task_id=None, meta_mode=None, mode='train', cached_output=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        bs = prev_output_tokens.shape[0]
    
        # Randomly initialized task embedding
        if self.training_mode == 'multitask':
            task_embedding = self.task_embeddings(task_id)
        elif 'meta' in self.training_mode:
            if mode == 'train':
                task_embeddings = self.task_embeddings
            else:
                task_embeddings = self.task_embeddings_eval
    
            if meta_mode == 'outer':
                task_embedding = task_embeddings(task_id).data
            else:
                task_embedding = task_embeddings(task_id)
        elif self.training_mode == 'single_task':
            task_embedding = self.task_embedding_init
        else:
            task_embedding = None
    
        if task_embedding is not None:
            if len(list(task_embedding.shape)) == 1:
                task_embedding = task_embedding.unsqueeze(0)
            if task_embedding.shape[0] == 1:
                task_embedding = task_embedding.expand(bs, -1)
    
            if self.task_emb_cond_type == 'encoder':
                task_embedding = task_embedding.unsqueeze(0)
                encoder_out = {
                    'encoder_out': task_embedding, # T x B x C
                    'encoder_padding_mask': torch.zeros(bs, 1, device=prev_output_tokens.device).bool() # B x T
                }
            else:
                assert self.task_emb_cond_type == 'decoder'
                task_embedding = task_embedding.unsqueeze(1)
                if self.task_emb_proj is not None:
                    task_embedding = self.task_emb_proj(task_embedding)

        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, task_embedding=task_embedding, cached_output=cached_output)
        x = self.output_layer(x)

        if task_embedding is not None and self.task_emb_cond_type == 'decoder':
            x = x[:, 1:]

        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, task_embedding=None, cached_output=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        if cached_output is not None:
            layer_idx = cached_output['layer_idx']
            x = cached_output['layer_output']

            # decoder layers
            for layer in self.layers[layer_idx + 1:]:
                x, attn = layer(
                    x,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                )

            if self.layer_norm:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

            return x, None

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        if task_embedding is not None and self.task_emb_cond_type == 'decoder':
            x = torch.cat([task_embedding, x], axis=1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer_idx, layer in enumerate(self.layers):
            if self.split_task_emb:
                assert encoder_out is not None
                num_layers = len(self.layers)
                split_shp = encoder_out['encoder_out'].shape[-1] // num_layers
                x, attn = layer(
                    x,
                    encoder_out['encoder_out'].split(split_shp, -1)[layer_idx],
                    encoder_out['encoder_padding_mask'],
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                )
            else:
                x, attn = layer(
                    x,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                )

            if layer_idx == self.freeze_bottom_layers:
                x = x.data

            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def extract_features_intermediate(self, layer_idx, layer_output, encoder_out=None, incremental_state=None, task_embedding=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        x = layer_output

        # decoder layers
        for layer in self.layers[layer_idx + 1:]:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        if self.LMinit: 
          
            if 'meta' in self.training_mode:

                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    for name, param in layer.named_parameters():
                        if 'encoder_attn' in name:
                            state_dict['decoder.layers.' + str(i) + '.' + name] = param.data
                state_dict['decoder.task_embeddings.weight'] = self.task_embeddings.weight.data
                state_dict['decoder.task_embeddings_eval.weight'] = self.task_embeddings.weight.data

            if (not self.share_input_output_embed) and ('decoder.embed_out' not in state_dict):
                state_dict['decoder.embed_out'] = state_dict['decoder.embed_tokens.weight']

        else:

            for k in list(state_dict.keys()):
                print(k)
                if "task_embedding" in k:
                    print('Ignoring: ', k)
                    del state_dict[k]

            if self.training_mode != 'task_agnostic':
                print('Note: Initializing task embedding with zeros')
                state_dict['decoder.task_embedding_init'] = torch.zeros(self.encoder_embed_dim)

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

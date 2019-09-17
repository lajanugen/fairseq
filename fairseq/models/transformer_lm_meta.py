from pdb import set_trace as bp
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_meta import (
    Embedding,
    TransformerDecoderMeta,
)
from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_lm_meta')
class TransformerLanguageModelMeta(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'transformer_lm.gbw.adaptive_huge': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2',
            'transformer_lm.wiki103.adaptive': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2',
            'transformer_lm.wmt19.en': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2',
            'transformer_lm.wmt19.de': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2',
            'transformer_lm.wmt19.ru': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2',
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--task-emb-size', type=int, metavar='N',
                            help='Task embedding size')

        parser.add_argument('--training_mode', default='multitask', type=str,
                            help='Multi-tasking/meta-learning')
        parser.add_argument('--regularization', action='store_true',
                            help='Enable/Disable all regularization')
        parser.add_argument('--max_tasks', default=16, type=int,
                            help='Maximum number of tasks.')
        parser.add_argument('--train_only_z', action='store_true',
                            help='Train only z.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = Embedding(len(task.source_dictionary), args.decoder_input_dim, task.source_dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerDecoderMeta(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=args.training_mode == 'task_agnostic'
        )

        if getattr(args, 'train_only_z', None) and args.train_only_z:
            assert args.training_mode == 'single_task'
            print("Model params are not tuned!")
            for param in decoder.parameters():
                param.requires_grad = False
            decoder.task_embedding_init.requires_grad = True

        return TransformerLanguageModelMeta(decoder)

    def upgrade_state_dict_named(self, state_dict, name):

        task_emb_size = None
        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                print('Ignoring: ', k)
                task_emb_size = state_dict[k].shape[-1]
                del state_dict[k]

        # if self.training_mode != 'task_agnostic':
        #     print('Note: Initializing task embedding with zeros')
        #     state_dict['task_embedding_init'] = torch.zeros(self.encoder_embed_dim)
        if task_emb_size:
            state_dict['decoder.task_embedding_init'] = torch.zeros(task_emb_size)

        return state_dict


@register_model_architecture('transformer_lm_meta', 'transformer_lm_meta')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)


@register_model_architecture('transformer_lm_meta', 'transformer_lm_meta_small')
def transformer_lm_meta_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.dropout = getattr(args, 'dropout', 0.0)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    # args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    # args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.0)
    # args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,40000')
    # args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '50265')
    base_lm_architecture(args)

@register_model_architecture('transformer_lm_meta', 'transformer_lm_meta_z32')
def transformer_lm_meta_z32(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 32)
    transformer_lm_meta_small(args)

@register_model_architecture('transformer_lm_meta', 'transformer_lm_meta_z8')
def transformer_lm_meta_z8(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 8)
    transformer_lm_meta_small(args)

@register_model_architecture('transformer_lm_meta', 'transformer_lm_meta_iwslt')
def transformer_lm_meta_iwslt(args):
    args.encoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    base_lm_architecture(args)

# @register_model_architecture('transformer_lm_meta', 'transformer_lm_big')
# def transformer_lm_big(args):
#     args.decoder_layers = getattr(args, 'decoder_layers', 12)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     base_lm_architecture(args)
# 
# 
# @register_model_architecture('transformer_lm_meta', 'transformer_lm_gpt2_small')
# def transformer_lm_gpt2_small(args):
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_layers = getattr(args, 'decoder_layers', 24)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.1)
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     args.activation_fn = getattr(args, 'activation_fn', 'gelu')
#     base_lm_architecture(args)

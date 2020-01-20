from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import choice
import higher

from fairseq import utils
from fairseq.models.classifier import Classifier

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_sentence_encoder_taskemb import init_bert_params
from fairseq.models.transformer_sentence_encoder_taskemb import TransformerSentenceEncoderTaskemb


# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.


def compute_accuracy(logits, targets, mask=None):
    predictions = logits.argmax(dim=1)
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    assert predictions.shape == targets.shape
    accuracy = predictions.eq(targets).float()
    if mask is not None:
        accuracy *= mask
        accuracy = accuracy.sum() / mask.sum()
    else:
        accuracy = accuracy.mean()

    return accuracy


def compute_loss(logits, target, normalize_loss=False, mask=None):
    logits = logits.view(-1, logits.size(-1))
    target = target.view(-1)

    if mask is not None:
        loss = F.cross_entropy(logits, target, reduction='none') * mask
        if normalize_loss:
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.sum()
    else:
        loss = F.cross_entropy(logits, target, reduction='mean' if normalize_loss else 'sum')
    return loss


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@register_model('classifier_maml')
class FairseqTransformerClassifierMaml(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--bias-kv', action='store_true',
                            help='if set, adding a learnable bias kv')
        parser.add_argument('--zero-attn', action='store_true',
                            help='if set, pads attn with zero')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                            ' and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N',
                            help='num segment in the input')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for pooler layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')

        parser.add_argument('--tune_model_params', action='store_true',
                            help='Tune model params')
        parser.add_argument('--encoder_type', default='transformer', type=str,
                            help='type of encoder: transformer/RNN')
        parser.add_argument('--task_emb_cond_type', default='token', type=str,
                            help='type of encoder: transformer/RNN')
        parser.add_argument('--training_mode', default='multitask', type=str,
                            help='Multi-tasking/meta-learning')
        parser.add_argument('--num_grad_updates', default=1, type=int,
                            help='Number of grad steps in inner loop')
        parser.add_argument('--meta_gradient', action='store_true',
                            help='Backprop through optimization or not')
        parser.add_argument('--regularization', action='store_true',
                            help='Enable/Disable all regularization')
        parser.add_argument('--normalize_loss', action='store_true',
                            help='Normalize loss by # examples')
        parser.add_argument('--use_momentum', action='store_true',
                            help='Use momentum for task embedding updates')
        parser.add_argument('--task_emb_size', default=128, type=int,
                            help='Size of task embedding')
        parser.add_argument('--log_losses', default=None, type=str,
                            help='Output z optimization losses')
        parser.add_argument('--z_lr', default=1e-3, type=float,
                            help='learning rate for optimizing z')
        parser.add_argument('--reinit_meta_opt', action='store_true',
                            help='Re-initialize meta opt for every step')
        parser.add_argument('--task_emb_init', default='mean', type=str,
                            help='How to initialize rask embedding.')
        parser.add_argument('--num_task_examples', default=100, type=int,
                            help='Number of examples in task description.')
        parser.add_argument('--encoder_layers', default=1, type=int,
                            help='Number of encoder layers.')
        parser.add_argument('--task_description_len', default=3, type=int,
                            help='Length of task description.')

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Return the wrapped version of the module
        return FairseqTransformerClassifierMaml(args, task)

    def __init__(self, args, task):
        super(FairseqTransformerClassifierMaml, self).__init__()

        dictionary = task.input_vocab
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.unk_index = dictionary.unk_index

        self.max_tasks = task.max_tasks
        self.num_train_tasks = task.num_train_tasks
        self.num_test_tasks = task.num_test_tasks
        self.max_seq_len = task.max_seq_len
        self.train_unseen_task = task.train_unseen_task
        self.task = task
        # self.task_embedding_inds = task.task_embedding_inds
        
        self.args = args
        self.training_mode = args.training_mode
        self.num_grad_updates = args.num_grad_updates
        self.normalize_loss = args.normalize_loss
        self.task_emb_size = args.task_emb_size
        self.log_losses = args.log_losses
        self.z_lr = args.z_lr

        # self.task_embeddings = nn.Embedding(64, self.task_emb_size * args.task_description_len)
        # self.task_embedding_init = nn.Embedding(1, self.task_emb_size * args.task_description_len)
        self.task_embeddings = nn.Embedding(64, self.task_emb_size)
        self.task_embedding_init = nn.Embedding(1, self.task_emb_size)
        self.task_embedding_init.weight.data.fill_(0)
        self.inner_opt = optim.Adam(self.task_embeddings.parameters(), lr=self.z_lr)
        self.init_z_optimizer = optim.Adam(self.task_embedding_init.parameters(), lr=1e-3)

        self.model = Classifier(args, task)

    def forward(
        self,
        src_tokens,
        src_lengths,
        targets,
        src_all_tokens=None,
        num_tasks=None,
        split_data=False,
        optimizer=None,
        mode='train'
    ):
        bs = src_tokens.shape[0]

        segment_labels = torch.zeros_like(src_tokens)

        task_id = src_tokens[:, 0]

        # Strip off task id
        src_tokens = src_tokens[:, 1:]

        outputs = {}

        if self.training_mode == 'maml_meta':

            num_tasks = self.task.sample_num_tasks
            num_ex_per_task = bs // num_tasks
            split_ratio = 0.5
            N_train = int(split_ratio * num_ex_per_task)
            train_mask = torch.cat((torch.ones(num_tasks, N_train), torch.zeros(num_tasks, num_ex_per_task - N_train)), dim=1).cuda()
            train_mask = train_mask.view(-1)
            test_mask = 1 - train_mask

            assert num_tasks <= 64
            task_id = torch.arange(num_tasks).view(-1, 1).repeat(1, num_ex_per_task).view(-1).cuda()

            if mode == 'train':
                self.init_z_optimizer.zero_grad()
                self.inner_opt.zero_grad()

            self.task_embeddings.weight.data.copy_(self.task_embedding_init.weight.data)
            with higher.innerloop_ctx(
                self.task_embeddings, self.inner_opt, copy_initial_weights=False
            ) as (femb, diffopt):
                for _ in range(self.num_grad_updates):
                    task_embedding = femb(task_id)
                    logits = self.model(
                        src_tokens,
                        task_embedding=task_embedding)

                    loss = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=train_mask)
                    diffopt.step(loss)

                task_embedding = femb(task_id)
                logits = self.model(
                    src_tokens,
                    task_embedding=task_embedding)
                loss = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=test_mask)

                # Moved to outer loop
                # loss.backward()
                # meta_grad = self.task_embeddings.weight.grad.sum(0)

            if self.task_embedding_init.weight.grad is None:
                self.task_embedding_init.weight.sum().backward() 

            # self.task_embedding_init.weight.grad.data.copy_(meta_grad)
            # if mode == 'train':
            #     self.init_z_optimizer.step()

            # task_embeddings = self.task_embeddings(task_id)
            # task_embeddings = task_embeddings.view(-1, task_len, self.task_emb_size)

            # task_ids_mask_inds = (task_len * torch.rand(bs, 1)).long()
            # task_ids_mask = torch.zeros(bs, task_len).scatter(1, task_ids_mask_inds, 1).unsqueeze(-1).cuda()

            # logits = self.model(src_tokens, task_embeddings) #, task_ids_mask=task_ids_mask)
            outputs['post_loss_train'] = loss

        else:

            task_embedding = self.task_embedding_init(torch.LongTensor([0]).cuda())
            # task_embedding = task_embedding.view(-1, task_len, self.task_emb_size)
            # task_ids_mask = compositional_task_ids.eq(self.unk_index).unsqueeze(-1).float()

            logits = self.model(src_tokens, task_embedding) # , task_ids_mask=task_ids_mask)

            outputs['post_loss_train'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss)

        outputs['post_accuracy_train'] = compute_accuracy(logits, targets)

        return outputs


@register_model_architecture('classifier_maml', 'cls_maml')
def toy_transformer_cls_maml(args):

    args.regularization = getattr(args, 'regularization', False)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    if not args.regularization:
        args.dropout = 0.0
        args.attention_dropout = 0.0
        args.act_dropout = 0.0

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.bias_kv = getattr(args, 'bias_kv', False)
    args.zero_attn = getattr(args, 'zero_attn', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.task_emb_size = getattr(args, 'task_emb_size', 128)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.apply_bert_init = getattr(args, 'apply_bert_init', False)

    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)

    args.tune_model_params = getattr(args, 'tune_model_params', False)
    args.meta_gradient = getattr(args, 'meta_gradient', False)
    args.use_momentum = getattr(args, 'use_momentum', False)
    args.reinit_meta_opt = getattr(args, 'reinit_meta_opt', False)

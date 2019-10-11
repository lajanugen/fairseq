from pdb import set_trace as bp
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fairseq import utils
# from fairseq.models.lmclassifier import LMClassifier

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_sentence_encoder_taskemb import init_bert_params
from fairseq.models.transformer_sentence_encoder_taskemb import TransformerSentenceEncoderTaskemb

if importlib.find_loader('higher') is not None:
    import higher

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.


def compute_loss(logits, target, mask=None, loss_mask=None):
    logits = logits.view(-1, logits.size(-1))
    target = target.contiguous().view(-1)

    assert logits.shape[0] == target.shape[0]

    if mask is not None:
        mask = mask.view(-1)

    if loss_mask is not None:
        loss_mask = loss_mask.view(-1)

    if (mask is not None) and (loss_mask is not None):
        overall_mask = mask * loss_mask
    elif mask is not None:
        overall_mask = mask
    elif loss_mask is not None:
        overall_mask = loss_mask
    else:
        overall_mask = None

    if overall_mask is not None:
        loss = F.cross_entropy(logits, target, reduction='none') * overall_mask
        loss = loss.sum()
    else:
        loss = F.cross_entropy(logits, target, reduction='sum')

    return loss


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pad_review_to_max_len(review, max_len, pad_symbol):
    review = review[:max_len]
    review = F.pad(review[:max_len], (0, max_len - len(review)), "constant", pad_symbol)
    return review


def subsequent_mask(size):
    "Mask out subsequent positions."
    return torch.triu(torch.Tensor(size, size).cuda().fill_(float('-inf')), 1)


def set_grads_flag(model, enable):
    for param in model.parameters():
        param.requires_grad = enable


class LMClassifier(nn.Module):

    def __init__(self, args, task):
        super(LMClassifier, self).__init__()

        dictionary = task.vocab
        self.vocab_size = dictionary.__len__()
        self.padding_idx = dictionary.pad()
        self.encoder_embed_dim = args.encoder_embed_dim
        self.training_mode = args.training_mode
        self.task_emb_size = args.task_emb_size

        self.num_grad_updates = args.num_grad_updates
        self.use_momentum = args.use_momentum
        self.log_losses = args.log_losses
        self.z_lr = args.z_lr

        self.sentence_encoder = TransformerSentenceEncoderTaskemb(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=args.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            add_bias_kv=args.bias_kv,
            add_zero_attn=args.zero_attn,
            task_emb_size=args.task_emb_size,
            task_emb_cond_type=args.task_emb_cond_type
        )

        self.classifier = nn.Linear(
            args.encoder_embed_dim, self.vocab_size)

        if not args.tune_model_params:
            print("Model params are not tuned!")
            set_grads_flag(self.sentence_encoder, False)
            set_grads_flag(self.classifier, False)

    def forward(self, src_tokens, task_embedding=None, attn_mask=None, segment_labels=None):

       segment_labels = torch.zeros_like(src_tokens)

       output, _ = self.sentence_encoder(
           src_tokens,
           segment_labels=segment_labels,
           task_embedding=task_embedding,
           self_attn_mask=attn_mask)

       output = output[-1].transpose(0, 1)

       if task_embedding is not None:
         # Ignore task embedding time-step
         output = output[:, 1:]
       rep_size = output.shape[-1]
       output = output.contiguous().view(-1, rep_size)

       logits = self.classifier(output)

       return logits


@register_model('lm_snail')
class FairseqReviewLM(BaseFairseqModel):

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
        parser.add_argument('--regularization', action='store_true',
                            help='Enable/Disable all regularization')
        parser.add_argument('--use_momentum', action='store_true',
                            help='Use momentum for task embedding updates')
        parser.add_argument('--task_emb_size', default=128, type=int,
                            help='Size of task embedding')
        parser.add_argument('--log_losses', default=None, type=str,
                            help='Output z optimization losses')
        parser.add_argument('--z_lr', default=1e-3, type=float,
                            help='learning rate for optimizing z')
        parser.add_argument('--encoder_layers', default=1, type=int,
                            help='Number of encoder layers.')
        parser.add_argument('--meta_num_ex', default=11, type=int,
                            help='Number of examples to use for meta learning.')
#        parser.add_argument('--max_seq_len', default=128, type=int,
#                            help='Maximum sequence length.')
#        parser.add_argument('--max_tasks', default=16, type=int,
#                            help='Maximum number of tasks.')

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Return the wrapped version of the module
        return FairseqReviewLM(args, task)

    def __init__(self, args, task):
        super(FairseqReviewLM, self).__init__()

        self.task = task
        dictionary = task.vocab
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_tasks = task.max_tasks
        self.encoder_type = args.encoder_type
        self.encoder_embed_dim = args.encoder_embed_dim
        self.training_mode = args.training_mode
        self.num_grad_updates = args.num_grad_updates
        self.use_momentum = args.use_momentum
        self.task_emb_size = args.task_emb_size
        self.log_losses = args.log_losses
        self.z_lr = args.z_lr
        self.num_train_tasks = task.num_train_tasks
        self.num_test_tasks = task.num_test_tasks
        self.max_seq_len = task.max_seq_len
        self.sample_num_tasks = task.sample_num_tasks
        self.meta_num_ex = args.meta_num_ex

        if self.training_mode == 'multitask':
            self.task_embeddings = nn.Embedding(
                # self.max_tasks, self.task_emb_size)
                self.num_train_tasks, self.task_emb_size)
            self.task_embeddings_eval = nn.Embedding(
                # self.max_tasks, self.task_emb_size)
                self.num_test_tasks, self.task_emb_size)

        elif 'meta' in self.training_mode:
            # Train
            self.task_embeddings = nn.Embedding(
                # self.max_tasks, self.task_emb_size)
                self.num_train_tasks, self.task_emb_size)
            self.z_optimizer = optim.Adam(
                self.task_embeddings.parameters(), lr=self.z_lr)
            # Eval
            self.task_embeddings_eval = nn.Embedding(
                # self.max_tasks, self.task_emb_size)
                self.num_test_tasks, self.task_emb_size)
            self.z_optimizer_eval = optim.Adam(
                self.task_embeddings_eval.parameters(), lr=self.z_lr)

        elif self.training_mode == 'single_task' or self.training_mode == 'maml_z':
            self.task_embedding_init = nn.Parameter(torch.randn(self.task_emb_size))

        self.model = LMClassifier(args, task)

        if self.training_mode == 'maml':
            self.inner_opt = optim.Adam(self.model.parameters(), lr=self.z_lr)
        elif self.training_mode == 'maml_z':
            self.inner_opt = optim.Adam(self.task_embedding_init, lr=self.z_lr)


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

        outputs = {}

        bs = src_tokens.shape[0]

        if not self.task.train_unseen_task:

            task_id = src_tokens[:, 0]
            src_tokens = src_tokens[:, 1:]

            num_tasks = self.sample_num_tasks

            num_ex_per_task = bs // num_tasks
    
            src_tokens = src_tokens.view(num_tasks, bs // num_tasks, -1)
            targets = targets.view(num_tasks, bs // num_tasks, -1)
            
            random_example_order = torch.LongTensor(np.random.permutation(num_ex_per_task))
    
            src_tokens = src_tokens[:, random_example_order]
            targets = targets[:, random_example_order]
    
            split_seq_len = self.meta_num_ex
            d = num_ex_per_task // split_seq_len
            num_ex_per_task = d * split_seq_len
    
            src_tokens = src_tokens[:, :num_ex_per_task].contiguous()
            targets = targets[:, :num_ex_per_task].contiguous()
    
            bs = num_tasks * num_ex_per_task
            new_bs = bs // split_seq_len
            src_tokens = src_tokens.view(new_bs, -1)
            seq_len = targets.shape[-1]
            targets = targets.view(-1, split_seq_len, seq_len)
            targets[:, :-1].fill_(self.task.vocab.pad())
            targets = targets.view(new_bs, -1)

            src_tokens = src_tokens[:, :-1]
            targets = targets[:, 1:]

            outputs['sample_size'] = targets.ne(self.task.vocab.pad()).sum().item()
        else:
            src_tokens = src_tokens[:, :-1]
            targets = targets[:, 1:]

        pad_mask = targets.eq(self.task.vocab.pad())
        loss_mask = 1 - pad_mask.float()
        loss_mask = loss_mask.cuda()

        if split_data:
            train_mask = torch.Tensor(targets.shape).uniform_(0, 2).long().float().cuda()
            test_mask = 1 - train_mask
        else:
            train_mask, test_mask = None, None

        if 'meta' in self.training_mode or self.training_mode == 'multitask':
            if mode == 'eval':
                task_embeddings = self.task_embeddings_eval
            else:
                task_embeddings = self.task_embeddings

        if self.training_mode == 'task_agnostic' or self.training_mode == 'maml':
            attn_mask = subsequent_mask(targets.shape[1])
        else:
            attn_mask = subsequent_mask(targets.shape[1] + 1)

        # Randomly initialized task embedding
        if self.training_mode == 'multitask':
            task_embedding = task_embeddings(task_id)
        elif self.training_mode == 'single_task' or self.training_mode == 'maml_z':
            task_embedding = self.task_embedding_init
        else:
            assert self.training_mode == 'task_agnostic'
            task_embedding = None

        if 'meta' in self.training_mode:

            if mode == 'eval':
                self.task_embeddings_eval.weight.data.zero_()
                z_optimizer = self.z_optimizer_eval
            else:
                self.task_embeddings.weight.data.zero_()
                z_optimizer = self.z_optimizer

            step_size = self.z_lr
            set_learning_rate(z_optimizer, step_size)

            losses = []
            for i in range(self.num_grad_updates):

                num_grad_updates = i

                z_optimizer.zero_grad()
                task_embedding = task_embeddings(task_id)

                logits = self.model(
                    src_tokens,
                    attn_mask=attn_mask,
                    task_embedding=task_embedding)

                loss = compute_loss(logits, targets, mask=train_mask, loss_mask=loss_mask)
                losses.append(loss.item())

                loss.backward()
                z_optimizer.step()

                if self.log_losses:
                    losses.append(loss.item())

                if i == 0:
                    outputs['pre_loss_train'] = compute_loss(logits, targets, mask=train_mask, loss_mask=loss_mask)
                    if split_data:
                        outputs['pre_loss_test'] = compute_loss(logits, targets, mask=test_mask, loss_mask=loss_mask)

                    prev_loss = loss.item()
                else:
                    cur_loss = loss.item()

                    if cur_loss > prev_loss:
                        step_size /= 2
                        set_learning_rate(z_optimizer, step_size)
                        if step_size < 1e-6:
                            break

                    prev_loss = cur_loss

            outputs['num_grad_updates'] = 1.0 * num_grad_updates
            if self.log_losses:
                if np.random.uniform() > 0.99:
                    with open(self.log_losses, 'a') as f:
                        losses_str = '%s\n' % ' '.join(map(str, losses))
                        f.write(losses_str)

        elif self.training_mode == 'maml' and mode == 'train':
            step_size = self.z_lr
            set_learning_rate(self.inner_opt, step_size)

            outputs['num_grad_updates'] = 1.0 * self.num_grad_updates

            with higher.innerloop_ctx(
                self.model, self.inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                for _ in range(self.num_grad_updates):
                    logits = fnet(
                        src_tokens,
                        attn_mask=attn_mask,
                        task_embedding=None)

                    loss = compute_loss(logits, targets, mask=train_mask, loss_mask=loss_mask)
                    diffopt.step(loss)

                logits = fnet(
                    src_tokens,
                    attn_mask=attn_mask,
                    task_embedding=None)

                loss = compute_loss(logits, targets, mask=test_mask, loss_mask=loss_mask)
                loss /= self.sample_num_tasks
                loss.backward()


        logits = self.model(
            src_tokens,
            attn_mask=attn_mask,
            task_embedding=task_embedding.data if ('meta' in self.training_mode or self.training_mode == 'maml_z') else task_embedding)

        outputs['post_loss_train'] = compute_loss(logits, targets, mask=train_mask, loss_mask=loss_mask)
        if 'pre_loss_train' in outputs:
            outputs['train_loss_delta'] = outputs['pre_loss_train'] - outputs['post_loss_train']
        if split_data:
            outputs['post_loss_test'] = compute_loss(logits, targets, mask=test_mask, loss_mask=loss_mask)
            if 'pre_loss_test' in outputs:
                outputs['test_loss_delta'] = outputs['pre_loss_test'] - outputs['post_loss_test']

        return outputs

    def upgrade_state_dict_named(self, state_dict, name):

        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                print('Ignoring: ', k)
                del state_dict[k]

        if self.training_mode != 'task_agnostic' and self.training_mode != 'maml':
            print('Note: Initializing task embedding with zeros')
            state_dict['task_embedding_init'] = torch.zeros(self.task_emb_size)

        return state_dict


@register_model_architecture('lm_snail', 'snail_tf')
def toy_transformer_cls(args):

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
    args.use_momentum = getattr(args, 'use_momentum', False)

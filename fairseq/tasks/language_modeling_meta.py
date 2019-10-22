from pdb import set_trace as bp
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import numpy as np

from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    MonolingualDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
    IndexedRawTextDataset,
    SubsetDataset
)
from fairseq.tasks import FairseqTask, register_task


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class IndexedRawTextDatasetCustom(IndexedRawTextDataset):

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, eval_mode=False, eval_task_id=0, complete_doc=False):
        self.eval_mode = eval_mode
        self.eval_task_id = eval_task_id
        self.complete_doc = complete_doc
        super().__init__(path, dictionary, append_eos=append_eos, reverse_order=reverse_order)

    def read_data(self, path, dictionary):
        if self.eval_mode:
            count = 0
        lines_list = []
        tokens_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    if self.eval_mode:
                        if count == self.eval_task_id - 1:
                            self.lines = []
                            self.tokens_list = []
                            self.sizes = []
                        elif count == self.eval_task_id:
                            break
                        count += 1
                    if self.complete_doc:
                        line = ' '.join(lines_list)
                        tokens = torch.cat(tokens_list, 0)

                        self.lines.append(line)
                        self.tokens_list.append(tokens)
                        self.sizes.append(len(tokens))

                        lines_list = []
                        tokens_list = []
                    continue

                if self.complete_doc:
                    line = '<s> ' + line
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()

                if self.complete_doc:
                    lines_list.append(line)
                    tokens_list.append(tokens)
                else:
                    self.lines.append(line)
                    self.tokens_list.append(tokens)
                    self.sizes.append(len(tokens))

        self.sizes = np.array(self.sizes)

@register_task("language_modeling_meta")
class LanguageModelingMetaTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        parser.add_argument('--train_unseen_task', action='store_true',
                            help='Train on unseen task')
        parser.add_argument('--eval_task_id', default=0, type=int,
                            help='Identifier of meta eval task')
        parser.add_argument('--no_training', action='store_true',
                            help='No fine-tuning.')
        parser.add_argument('--z_lr', default=1e-3, type=float,
                            help='learning rate for optimizing z')
        parser.add_argument('--num_grad_updates', default=1, type=int,
                            help='Number of grad steps in inner loop')
        parser.add_argument('--LMinit', action='store_true',
                            help='LM init.')
        parser.add_argument('--mdl', default='ff', type=str,
                            help='Model type.')
        parser.add_argument('--task_emb_layer', default=-1, type=int,
                            help='Layer at which task embedding is inserted.')
        parser.add_argument('--freeze_bottom_layers', default=-1, type=int,
                            help='Number of bottom layers to freeze.')
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

        self.no_training = args.no_training
        self.eval_task_id = args.eval_task_id
        self.train_unseen_task = args.train_unseen_task
        self.dataset_size = {}
        self.z_lr = args.z_lr
        self.num_grad_updates = args.num_grad_updates
        self.mdl = args.mdl
        self.task_emb_layer = args.task_emb_layer
        self.freeze_bottom_layers = args.freeze_bottom_layers

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        if getattr(args, "raw_text", False):
            utils.deprecation_warning(
                "--raw-text is deprecated, please use --dataset-impl=raw"
            )
            args.dataset_impl = "raw"
        elif getattr(args, "lazy_load", False):
            utils.deprecation_warning(
                "--lazy-load is deprecated, please use --dataset-impl=lazy"
            )
            args.dataset_impl = "lazy"

        dictionary = None
        output_dictionary = None
        if args.data:
            paths = args.data.split(":")
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            print("| dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0

        data_path = paths[epoch % len(paths)]

        if self.train_unseen_task:
            split_path = os.path.join(data_path, 'test.' + split)
        else:
            split_path = os.path.join(data_path, 'train.' + split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )

        if self.train_unseen_task:
          assert self.args.sample_break_mode != 'complete_doc'
          self.dataset_size[split] = len(dataset)
          if ('train' in self.dataset_size) and ('valid' in self.dataset_size):
            assert self.dataset_size['train'] == self.dataset_size['valid']
          dataset = SubsetDataset(dataset, self.eval_task_id)
          print("Data size %s: %d" % (split, len(dataset)))

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode="eos",
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
                add_bos_token=self.args.add_bos_token,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    def split_doc(self, sample):
        src_reviews = []
        tgt_reviews = []
        task_id = []
        count = 0

        dim2 = sample['net_input']['src_tokens'].shape[1]

        src = sample['net_input']['src_tokens'].view(-1).cpu()
        tgt = sample['target'].view(-1).cpu()

        review_boundaries = src.eq(self.dictionary.eos()).nonzero()

        src_split = np.split(src, review_boundaries)
        assert src_split[0].numel() == 0
        src_reviews = src_split[1:]

        tgt_split = np.split(tgt, review_boundaries)
        assert tgt_split[0].numel() == 0
        tgt_reviews = tgt_split[1:]

        task_id = (review_boundaries/dim2).long().view(-1).cuda()

        # for i in range(sample['target'].shape[0]):
        #     src = sample['net_input']['src_tokens'][i].cpu()
        #     tgt = sample['target'][i].cpu()

        #     review_boundaries = src.eq(self.dictionary.eos()).nonzero()

        #     src_split = np.split(src, review_boundaries)
        #     assert src_split[0].numel() == 0
        #     src_split = src_split[1:]
        #     src_reviews.extend(src_split)

        #     tgt_split = np.split(tgt, review_boundaries)
        #     assert tgt_split[0].numel() == 0
        #     tgt_split = tgt_split[1:]
        #     tgt_reviews.extend(tgt_split)

        #     task_id.extend([count]*len(src_split))
        #     count += 1

        # task_id = torch.LongTensor(task_id).cuda()

        src_tokens = data_utils.collate_tokens(src_reviews, self.dictionary.pad()).cuda()
        target = data_utils.collate_tokens(tgt_reviews, self.dictionary.pad()).cuda()
    
        assert task_id.shape[0] == src_tokens.shape[0]

        sample['net_input']['src_tokens'] = src_tokens
        sample['target'] = target
        sample['net_input']['task_id'] = task_id

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        sample['net_input']['mode'] = 'train'
        
        if (self.args.sample_break_mode == 'complete_doc') and (self.mdl != 'snail'):
            self.split_doc(sample)

        if 'meta' in model.decoder.training_mode:
            model.decoder.task_embeddings.weight.data.zero_()
            z_optimizer = model.decoder.z_optimizer

            step_size = self.z_lr
            # set_learning_rate(z_optimizer, step_size)

            if self.task_emb_layer >= 0:
                _, inner_states = model(**sample['net_input'])
                inner_outputs = inner_states['inner_states'][self.task_emb_layer].data
                cached_output = {'layer_idx': self.task_emb_layer, 'layer_output': inner_outputs}

            sample['net_input']['meta_mode'] = 'inner'
            if self.task_emb_layer >= 0:
                sample['net_input']['cached_output'] = cached_output

            for param in model.decoder.parameters():
                param.requires_grad = False
            model.decoder.task_embeddings.weight.requires_grad = True

            num_grad_updates = self.num_grad_updates
            best_loss = float('inf')
            for i in range(self.num_grad_updates):

                z_optimizer.zero_grad()

                loss, sample_size, logging_output = criterion(model, sample)

                loss.backward()
                z_optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_embeddings = model.decoder.task_embeddings.weight.data
                    num_grad_updates = i
            model.decoder.task_embeddings.weight.data.copy_(best_embeddings)

        for param in model.decoder.parameters():
            param.requires_grad = True
        sample['net_input']['meta_mode'] = 'outer'
        optimizer.zero_grad()
        if self.freeze_bottom_layers >= 0:
            sample['net_input']['cached_output'] = None
        loss, sample_size, logging_output = criterion(model, sample)
        if 'meta' in model.decoder.training_mode:
            logging_output['num_grad_updates'] = num_grad_updates
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        sample['net_input']['mode'] = 'eval'

        if (self.args.sample_break_mode == 'complete_doc') and (self.mdl != 'snail'):
            self.split_doc(sample)

        bs = sample['target'].shape[0]
        # train_mask = (torch.FloatTensor(bs, 1).uniform_() > 0.5).float().cuda()
        # test_mask = 1 - train_mask

        # We need gradient computation
        if 'meta' in model.decoder.training_mode:

            with torch.set_grad_enabled(True):

                model.decoder.task_embeddings_eval.weight.data.zero_()
                z_optimizer = model.decoder.z_optimizer_eval
                # set_learning_rate(z_optimizer, self.z_lr)

                for i in range(self.num_grad_updates):

                    z_optimizer.zero_grad()
                    sample['net_input']['meta_mode'] = 'inner'
                    # loss, sample_size, _ = criterion(model, sample, reduce=False)
                    # train_loss = (loss.view(bs, -1) * train_mask).sum()
                    # train_loss.backward()
                    loss, sample_size, _ = criterion(model, sample)
                    loss.backward()
                    z_optimizer.step()

        loss, sample_size, logging_output = criterion(model, sample)
        if 'meta' in model.decoder.training_mode:
            logging_output['num_grad_updates'] = self.num_grad_updates

        # if self.train_unseen_task:
        #     loss, sample_size, logging_output = criterion(model, sample)
        # else:
        #     loss, sample_size, logging_output = criterion(model, sample, reduce=False)
        #     test_loss = (loss.view(bs, -1) * test_mask).sum()
        #     logging_output['loss'] = test_loss
        #     for key in ['ntokens', 'nsentences', 'sample_size']:
        #         logging_output[key] = logging_output[key]/2

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        for other_metrics in ['num_grad_updates']:
            agg_logging_outputs[other_metrics] = sum(
                log[other_metrics] for log in logging_outputs if other_metrics in log)
        return agg_logging_outputs 

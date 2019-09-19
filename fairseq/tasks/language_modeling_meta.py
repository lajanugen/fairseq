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
    IndexedRawTextDataset
)
from fairseq.tasks import FairseqTask, register_task


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class IndexedRawTextDatasetCustom(IndexedRawTextDataset):

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, mode=None):
        super().__init__(path, dictionary, append_eos=append_eos, reverse_order=reverse_order)
        self.mode = mode

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    continue
                self.lines.append(line)
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
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

        # dataset = data_utils.load_indexed_dataset(
        #     split_path, self.dictionary, self.args.dataset_impl, combine=combine
        # )
        dataset = IndexedRawTextDatasetCustom(split_path, self.dictionary)
        print('| loaded {} examples from: {}'.format(len(dataset), split_path))
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        if self.train_unseen_task:
          self.dataset_size[split] = len(dataset)
          print("Data size %s: %d" % (split, len(dataset)))
          if ('train' in self.dataset_size) and ('valid' in self.dataset_size):
            assert self.dataset_size['train'] == self.dataset_size['valid']
          dataset = SubsetDataset(dataset, self.eval_task_id)

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

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()

        if 'meta' in model.decoder.training_mode:
            model.decoder.task_embeddings.weight.data.zero_()
            z_optimizer = model.decoder.z_optimizer

            step_size = self.z_lr
            set_learning_rate(z_optimizer, step_size)

            for i in range(self.num_grad_updates):

                z_optimizer.zero_grad()
                sample['net_input']['meta_mode'] = 'inner'
                loss, sample_size, logging_output = criterion(model, sample)

                loss.backward()
                z_optimizer.step()

                if i == 0:
                    # outputs['pre_loss_train'] = compute_loss(logits, targets, mask=train_mask, loss_mask=loss_mask)
                    # if split_data:
                    #     outputs['pre_loss_test'] = compute_loss(logits, targets, mask=test_mask, loss_mask=loss_mask)

                    prev_loss = loss.item()
                else:
                    cur_loss = loss.item()

                    if cur_loss > prev_loss:
                        step_size /= 2
                        set_learning_rate(z_optimizer, step_size)
                        if step_size < 1e-6:
                            break

                    prev_loss = cur_loss

        sample['net_input']['meta_mode'] = 'outer'
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        sample['net_input']['mode'] = 'eval'
        if 'meta' in model.decoder.training_mode:
            with torch.set_grad_enabled(True):
                model.decoder.task_embeddings.weight.data.zero_()
                z_optimizer = model.decoder.z_optimizer
                set_learning_rate(z_optimizer, self.z_lr)

                for i in range(self.num_grad_updates):

                    z_optimizer.zero_grad()
                    sample['net_input']['meta'] = 'inner'
                    loss, _, _ = criterion(model, sample)
                    loss.backward()
                    z_optimizer.step()

            loss, sample_size, logging_output = criterion(model, sample)
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = criterion(model, sample)

        return loss, sample_size, logging_output


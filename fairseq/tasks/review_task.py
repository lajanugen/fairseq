from pdb import set_trace as bp
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    PadDataset,
    SortDataset,
    TokenBlockDataset,
    SubsetDataset
)
from fairseq.tasks import FairseqTask, register_task


@register_task('review_task')
class ReviewTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of tokens per sample')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>) to each sample')

        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--train_unseen_task', action='store_true',
                            help='Train on unseen task')
        parser.add_argument('--eval_task_id', default=0, type=int,
                            help='Identifier of meta eval task')
        parser.add_argument('--no_training', action='store_true',
                            help='No fine-tuning.')

        # fmt: on

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.no_training = args.no_training
        self.eval_task_id = args.eval_task_id
        self.train_unseen_task = args.train_unseen_task
        self.dataset_size = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0

        vocab = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # e.g., /path/to/data/train.{bin,idx}
        if self.train_unseen_task:
            split_path = os.path.join(data_path, 'test.' + split)
        else:
            split_path = os.path.join(data_path, 'train.' + split + '.shuf')

        dataset = data_utils.load_indexed_dataset(split_path, self.vocab, self.args.dataset_impl)
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        if self.train_unseen_task:
          self.dataset_size[split] = len(dataset)
          print("Data size %s: %d" % (split, len(dataset)))
          if ('train' in self.dataset_size) and ('valid' in self.dataset_size):
            assert self.dataset_size['train'] == self.dataset_size['valid']
          dataset = SubsetDataset(dataset, self.eval_task_id)

        # prepend a beginning of sentence token (<s>) to each sample
        if self.args.add_bos_token:
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # we predict the next word, so roll the inputs by 1 position
        # input_tokens = RollDataset(dataset, shifts=1)
        # target_tokens = dataset
        input_tokens = dataset

        # define the structure of each batch. "net_input" is passed to the
        # model's forward, while the full sample (including "target") is passed
        # to the criterion
        dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': PadDataset(
                        input_tokens,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False
                    ),
                    'src_lengths': NumelDataset(input_tokens, reduce=False),
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(input_tokens, reduce=True),
            },
            sizes=[input_tokens.sizes],
        )

        # shuffle the dataset and then sort by size
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
            dataset = SortDataset(
                dataset,
                sort_order=[
                    shuffle,
                    input_tokens.sizes,
                ],
            )

        self.datasets[split] = dataset

    def _get_loss(self, sample, model, criterion, split_data=False):

        sample['net_input']['split_data'] = split_data

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']
        outputs['loss'] = loss

        sample_size = sample['ntokens']

        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample['ntokens'],
        }

        self.logging_diagnostics = outputs.keys()

        for diagnostic in outputs:
            value = outputs[diagnostic]
            if type(value) == torch.Tensor:
                value = value.item()
            logging_output[diagnostic] = value

        # loss, sample_size, logging_output = criterion(model, sample)

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        return agg_logging_outputs

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        optimizer.zero_grad()
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        sample['net_input']['mode'] = 'eval'
        if 'meta' in model.training_mode:
            with torch.set_grad_enabled(True):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
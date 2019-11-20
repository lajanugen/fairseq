# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from collections import OrderedDict

import numpy as np
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import register_task
from fairseq.tasks.synthetic_task import SyntheticLMTask
from fairseq.tasks.synthetic_task_generator import TaskGenerator

@register_task('synthetic_lm_task_maml')
class SyntheticLMMAMLTask(SyntheticLMTask):

    def __init__(self, args, vocab, load_data=True):
        super().__init__(args, vocab, load_data)

    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample_size = targets.ne(self.vocab.pad()).sum().item()

        logging_output = {
            'ntokens': sample_size,
            'sample_size': sample_size,
        }
        loss = 0

        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']

        sample['net_input']['split_data'] = split_data

        task_ids = src_tokens[:, 0].unique()
        task_num = task_ids.numel()

        for i in range(task_num):
            sample_ids = (src_tokens[:, 0] == task_ids[i])
            sample['net_input']['src_tokens'] = src_tokens[sample_ids, :]
            sample['net_input']['src_lengths'] = src_lengths[sample_ids]
            sample['net_input']['targets'] = targets[sample_ids, :]
        
            outputs = model(**sample['net_input'])

            loss += outputs['post_loss_train']
            outputs['loss'] = loss

            self.logging_diagnostics = outputs.keys()

        for diagnostic in outputs:
            value = outputs[diagnostic]
            if type(value) == torch.Tensor:
                value = value.item()
            logging_output[diagnostic] = value

        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        optimizer.zero_grad()
        sample['net_input']['mode'] = 'train'

        loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.step()

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        sample['net_input']['mode'] = 'eval'
        # skip finetuning on valid tasks, since we'll run grid-search over multiple checkpoints,
        # and in test time we will train on the same task first
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        return loss, sample_size, logging_output



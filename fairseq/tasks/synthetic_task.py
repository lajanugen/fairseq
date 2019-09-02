# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import register_task
from fairseq.tasks.review_task import ReviewTask
from fairseq.tasks.synthetic_task_generator import TaskGenerator


@register_task('synthetic_lm_task')
class SyntheticLMTask(ReviewTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('--load_tasks_file', default='', type=str,
                            help='Tasks file.')
        parser.add_argument('--max_tasks', default=5, type=int,
                            help='Max tasks to precompute')
        parser.add_argument('--max_seq_len', default=16, type=int,
                            help='Maximum sequence length')

    @classmethod
    def setup_task(cls, args, **kwargs):
        vocab = Dictionary()
        for v in range(args.vocab_size)
            vocab.add_symbol(v)
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        self.max_tasks = args.max_tasks
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.sample_num_tasks = args.sample_num_tasks

        task_generator = TaskGenerator(
            self.max_tasks,
            self.num_train,
            self.max_seq_len,
            self.vocab_size)
        task_descriptions = task_generator.load_tasks(args.load_tasks_file)

        assert len(task_descriptions) >= self.num_train_tasks + 2 * self.num_test_tasks

        if self.train_unseen_task:
            test_task_descriptions = task_descriptions[-self.num_test_tasks:]

            test_tasks = task_generator.generate_data(
                test_task_descriptions, self.num_train, self.num_test)

            train_examples = [task[0] for task in test_tasks]
            val_examples = [task[1] for task in test_tasks]
            test_examples = [task[2] for task in test_tasks]

            self.examples = {'train': train_examples, 'valid': val_examples, 'test': test_examples}

        else:
            train_task_descriptions = task_descriptions[:self.num_train_tasks]
            val_task_descriptions = task_descriptions[self.num_train_tasks : self.num_train_tasks + self.num_test_tasks]

            print('Generating data...')
            train_tasks = task_generator.generate_data(
                train_task_descriptions, self.num_train, self.num_test)
            val_tasks = task_generator.generate_data(
                val_task_descriptions, self.num_train, self.num_test)
            print('Done Generating data.')

            train_examples = [task[0] for task in train_tasks]
            val_examples = [task[0] for task in val_tasks]

            self.examples = {'train': train_examples, 'valid': val_examples}


    def construct_data(self, examples):

        input_sentences, output_sentences, lengths = [], [], [] 

        for instance in examples:
            orig_seq, transform_seq = instance

            assert len(orig_seq) == self.max_seq_len
            assert len(transform_seq) == self.max_seq_len

            orig_seq_enc = [self.index(s) for s in input_seq]
            transform_seq_enc = [self.index(s) for s in input_seq]

            input_sentence = orig_seq_enc + transform_seq_enc + [self.vocab.eos()]
            output_sentence = [self.vocab.pad()] * len(orig_seq_enc) + transform_seq_enc + [self.vocab.eos()]
 
            # prepend a beginning of sentence token (<s>) to each sample
            if self.args.add_bos_token:
                input_sequence = [self.vocab.bos()] + input_sequence
                output_sequence = [self.vocab.pad()] + output_sequence

            assert len(input_sequence) == len(output_sequence)
           
            input_sentences.append(torch.LongTensor(input_sequence))
            output_sentences.append(torch.LongTensor(output_sequence))
            lengths.append(len(input_sequence))

        return input_sentences, output_sentences, lengths


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if self.train_unseen_task:
            assert self.eval_task_id < self.num_test_tasks

            input_sentences, output_sentences, lengths = self.construct_data(examples[split][task_id])

            self.datasets[split] = LanguagePairDataset(
                src=input_sentences,
                src_sizes=lengths,
                src_dict=self.vocab,
                tgt=output_sentences,
                tgt_sizes=lengths,
                tgt_dict=self.vocab,
                left_pad_source=False,
                max_target_positions=lengths[0],
                input_feeding=False,
            )

        else:
            dataset_map = OrderedDict()
            split_examples = examples[split]
            num_tasks = len(split_examples)

            for task_id in range(num_tasks):
                input_sentences, output_sentences, lengths = self.construct_data(split_examples[task_id])

                dataset_map[task_id] = LanguagePairDataset(
                    src=input_sentences,
                    src_sizes=lengths,
                    src_dict=self.vocab,
                    tgt=output_sentences,
                    tgt_sizes=lengths,
                    tgt_dict=self.vocab,
                    left_pad_source=False,
                    max_target_positions=lengths[0],
                    input_feeding=False,
                )
            self.datasets[split] = MultiCorpusSampledDataset(
                dataset_map, num_samples=self.sample_num_tasks)


    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['split_data'] = split_data
        sample['net_input']['num_tasks'] = self.sample_num_tasks

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']
        outputs['nll_loss'] = loss

        sample_size = sample['nsentences']

        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample['target'].size(0),
        }

        self.logging_diagnostics = outputs.keys()

        for diagnostic in outputs:
            value = outputs[diagnostic]
            if type(value) == torch.Tensor:
                value = value.item()
            logging_output[diagnostic] = value

        return loss, sample_size, logging_output



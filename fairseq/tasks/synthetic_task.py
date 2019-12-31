from pdb import set_trace as bp
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from collections import OrderedDict

import numpy as np
import torch
import pickle

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import register_task
from fairseq.tasks.review_task import ReviewTask
from fairseq.tasks.synthetic_task_generator import TaskGenerator


@register_task('synthetic_lm_task')
class SyntheticLMTask(ReviewTask):

    @staticmethod
    def add_args(parser):
        super(SyntheticLMTask, SyntheticLMTask).add_args(parser)
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('--max_tasks', default=5, type=int,
                            help='Max tasks to precompute')
        parser.add_argument('--max_seq_len', default=16, type=int,
                            help='Maximum sequence length')
        parser.add_argument('--load_train_tasks_file', default='/checkpoint/llajan/tasks.txt', type=str,
                            help='Train tasks file.')
        parser.add_argument('--load_test_tasks_file', default='/checkpoint/llajan/tasks.txt', type=str,
                            help='Test tasks file.')
        parser.add_argument('--load_test_id_file', default='', type=str,
                            help='Test unseen stage id file.')
        parser.add_argument('--unseen_task_stage', default=-1, type=int,
                            help='The operation to be inferred during test')
        parser.add_argument('--load_tasks_file_folder', default='', type=str,
                            help='Tasks file base directory')
        parser.add_argument('--load_from_pickle', action='store_true',
                            help='load tasks and examples from a pickle file')
        parser.add_argument('--vocab_size', default=10, type=int,
                            help='Vocabulary size')
        parser.add_argument('--num_train_tasks', default=5, type=int,
                            help='Number of train tasks')
        parser.add_argument('--num_test_tasks', default=5, type=int,
                            help='Number of test tasks')
        parser.add_argument('--num_train', default=10000, type=int,
                            help='Num training examples')
        parser.add_argument('--num_test', default=10000, type=int,
                            help='Num test examples')
        parser.add_argument('--sample_num_tasks', default=1, type=int,
                            help='Num of tasks to sample for each iteration')
        parser.add_argument('--num_operations', default=10000, type=int,
                            help='Total number of operations from each stage')


    @classmethod
    def setup_task(cls, args, load_data=True, **kwargs):
        vocab = Dictionary()
        for v in range(args.vocab_size):
            vocab.add_symbol(str(v))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab, load_data)

    def __init__(self, args, vocab, load_data=True):
        super().__init__(args, vocab)

        self.max_tasks = args.max_tasks
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.sample_num_tasks = args.sample_num_tasks
        self.load_from_pickle = args.load_from_pickle
        self.unseen_task_stage = args.unseen_task_stage

        self.train_task_ids = []
        self.val_task_ids = []
        self.test_task_ids = []

        self.num_operations = args.num_operations

        if self.load_from_pickle:
            if self.train_unseen_task:
                f = open(args.load_test_tasks_file, 'rb')
                data = pickle.load(f)
                f.close()

                test_tasks = []
                self.test_task_ids = []
                for dat in data:
                    self.test_task_ids.append(dat[1])

                    assert len(dat[0]) >= self.num_train + 2 * self.num_test
                    train = dat[0][:self.num_train]
                    val = dat[0][self.num_train:self.num_train+self.num_test]
                    test = dat[0][-self.num_test:]
                
                    test_tasks.append((train, val, test))
            else:
                f = open(args.load_train_tasks_file, 'rb')
                data = pickle.load(f)
                f.close()
             
                assert len(data) >= self.num_train_tasks+self.num_test_tasks
                data = data[:self.num_train_tasks+self.num_test_tasks]
               
                train_tasks = []
                self.train_task_ids = []
                for dat in data[:-self.num_test_tasks]:
                    self.train_task_ids.append(dat[1])

                    assert len(dat[0]) >= self.num_train + 2 * self.num_test
                    train = dat[0][:self.num_train]
                    val = dat[0][self.num_train:self.num_train+self.num_test]
                    test = dat[0][-self.num_test:]
               
                    train_tasks.append((train, val, test))

                val_tasks = []
                self.val_task_ids = []
                for dat in data[-self.num_test_tasks:]:
                    self.val_task_ids.append(dat[1])

                    assert len(dat[0]) >= self.num_train + 2 * self.num_test
                    train = dat[0][:self.num_train]
                    val = dat[0][self.num_train:self.num_train+self.num_test]
                    test = dat[0][-self.num_test:]
               
                    val_tasks.append((train, val, test))
        else:
            task_generator = TaskGenerator(
                self.max_tasks,
                self.max_seq_len,
                self.vocab_size)

            print('Generating data...')
            
            if load_data:
                if self.train_unseen_task:
                    test_task_descriptions = task_generator.load_tasks(args.load_test_tasks_file) 
                    self.test_task_ids = [list(task_generator.get_task_ids(t)) for t in test_task_descriptions]

                    test_tasks = task_generator.generate_data(
                        test_task_descriptions, self.num_train, self.num_test)
                else:
                    train_task_descriptions_full = task_generator.load_tasks(args.load_train_tasks_file)

                    assert len(train_task_descriptions_full) > self.num_train_tasks+self.num_test_tasks
                    train_task_descriptions_full = train_task_descriptions_full[:self.num_train_tasks+self.num_test_tasks]
                    
                    train_task_descriptions = train_task_descriptions_full[:-self.num_test_tasks]
                    val_task_descriptions = train_task_descriptions_full[-self.num_test_tasks:]
            
                    self.train_task_ids = [list(task_generator.get_task_ids(t)) for t in train_task_descriptions]
                    self.val_task_ids = [list(task_generator.get_task_ids(t)) for t in val_task_descriptions]

                    train_tasks = task_generator.generate_data(
                        train_task_descriptions, self.num_train, self.num_test)
                    val_tasks = task_generator.generate_data(
                        val_task_descriptions, self.num_train, self.num_test)
            
            print('Done Generating data.')

            self.num_operations = len(task_generator.transforms) + len(task_generator.subseq) + len(task_generator.reorder)

        if self.train_unseen_task:
            if load_data:
                if os.path.exists(args.load_test_id_file):
                    with open(args.load_test_id_file, 'r') as f:
                        unseen_stage_ids = [int(line.strip()) for line in f]
                    assert len(unseen_stage_ids) == len(test_tasks)

                    train_examples = []
                    for i, task in enumerate(test_tasks):
                        single_task_examples = []
                        for ex in task[0]:
                            single_task_examples.append((ex[0], ex[1], unseen_stage_ids[i]))
                        train_examples.append(single_task_examples)

                    val_examples = []
                    for i, task in enumerate(test_tasks):
                        single_task_examples = []
                        for ex in task[1]:
                            single_task_examples.append((ex[0], ex[1], unseen_stage_ids[i]))
                        val_examples.append(single_task_examples)

                    test_examples = []
                    for i, task in enumerate(test_tasks):
                        single_task_examples = []
                        for ex in task[2]:
                            single_task_examples.append((ex[0], ex[1], unseen_stage_ids[i]))
                        test_examples.append(single_task_examples)

                else:
                    train_examples = [task[0] for task in test_tasks]
                    val_examples = [task[1] for task in test_tasks]
                    test_examples = [task[2] for task in test_tasks]

                self.examples = {'train': train_examples, 'valid': val_examples, 'test': test_examples}

        else:
            train_examples = [task[0] for task in train_tasks]
            val_examples = [task[0] for task in val_tasks]

            self.examples = {'train': train_examples, 'valid': val_examples}


    def construct_data(self, task_ids, examples):

        input_sentences, output_sentences, src_lengths, tgt_lengths = [], [], [], []

        for instance in examples:
            if len(instance) == 2:
                orig_seq, transform_seq = instance
                unseen_id = self.unseen_task_stage
            elif len(instance) == 3:
                orig_seq, transform_seq, unseen_id = instance
            else:
                print('invalid data!')
                print(instance)

            if not self.load_from_pickle:
                assert len(orig_seq) == self.max_seq_len
                assert len(transform_seq) == self.max_seq_len

            orig_seq = map(str, orig_seq)
            transform_seq = map(str, transform_seq)
            
            orig_seq_enc = [self.vocab.index(s) for s in orig_seq]
            transform_seq_enc = [self.vocab.index(s) for s in transform_seq]

            input_sequence = orig_seq_enc + transform_seq_enc + [self.vocab.eos()]
            output_sequence = [self.vocab.pad()] * len(orig_seq_enc) + transform_seq_enc + [self.vocab.eos()]
 
            # prepend a beginning of sentence token (<s>) to each sample
            if self.args.add_bos_token:
                input_sequence = [self.vocab.bos()] + input_sequence
                output_sequence = [self.vocab.pad()] + output_sequence

            assert len(input_sequence) == len(output_sequence)

            # if loaded from pickle, the output might be of variant lengths.
            # max_seq_len determines the maximum final length we want to keep
            if self.load_from_pickle and len(input_sequence) < self.max_seq_len:
                pad_len = self.max_seq_len - len(input_sequence)
                input_sequence += [self.vocab.pad()] * pad_len
                output_sequence += [self.vocab.pad()] * pad_len
            elif self.load_from_pickle and len(input_sequence) > self.max_seq_len:
                input_sequence = input_sequence[:self.max_seq_len]
                output_sequence = output_sequence[:self.max_seq_len]

            # prepend task_id and unseen_stage_id
            input_sequence = task_ids + [unseen_id] + input_sequence
            # output_sequence = [task_id] + output_sequence
           
            input_sentences.append(torch.LongTensor(input_sequence))
            output_sentences.append(torch.LongTensor(output_sequence))
            src_lengths.append(len(input_sequence))
            tgt_lengths.append(len(output_sequence))

        return input_sentences, output_sentences, src_lengths, tgt_lengths


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if self.train_unseen_task:
            assert self.eval_task_id < len(self.examples[split])

            input_sentences, output_sentences, src_lengths, tgt_lengths = self.construct_data([self.eval_task_id] + self.test_task_ids[self.eval_task_id], self.examples[split][self.eval_task_id])

            self.datasets[split] = LanguagePairDataset(
                src=input_sentences,
                src_sizes=src_lengths,
                src_dict=self.vocab,
                tgt=output_sentences,
                tgt_sizes=tgt_lengths,
                tgt_dict=self.vocab,
                left_pad_source=False,
                max_target_positions=tgt_lengths[0],
                input_feeding=False,
            )

        else:
            dataset_map = OrderedDict()
            split_examples = self.examples[split]
            num_tasks = len(split_examples)
            if split == 'train':
                task_ids = self.train_task_ids
            elif split == 'valid':
                task_ids = self.val_task_ids
            
            for task_id in range(num_tasks):
                input_sentences, output_sentences, src_lengths, tgt_lengths = self.construct_data([task_id] + task_ids[task_id], split_examples[task_id])

                dataset_map[task_id] = LanguagePairDataset(
                    src=input_sentences,
                    src_sizes=src_lengths,
                    src_dict=self.vocab,
                    tgt=output_sentences,
                    tgt_sizes=tgt_lengths,
                    tgt_dict=self.vocab,
                    left_pad_source=False,
                    max_target_positions=tgt_lengths[0],
                    input_feeding=False,
                )
            self.datasets[split] = MultiCorpusSampledDataset(
                dataset_map, num_samples=self.sample_num_tasks)


    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['split_data'] = split_data

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']
        outputs['loss'] = loss

        # only count the length of the actual target sequence, not including the input sequence
        # sample_size = targets.numel() - targets.eq(self.vocab.pad()).sum().item()  # sample['ntokens']
        sample_size = targets.ne(self.vocab.pad()).sum().item()

        logging_output = {
            'ntokens': sample_size,
            'sample_size': sample_size,
        }

        self.logging_diagnostics = outputs.keys()

        for diagnostic in outputs:
            value = outputs[diagnostic]
            if type(value) == torch.Tensor:
                value = value.item()
            logging_output[diagnostic] = value

        return loss, sample_size, logging_output



from collections import OrderedDict
from scipy import stats
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import FairseqTask, register_task
# from fairseq.tasks.task_generator_v2 import TaskGenerator
from fairseq.tasks.task_generator import TaskGenerator

import numpy as np
# from pdb import set_trace as bp

class Dictionary_toy(Dictionary):
    def __init__(self, no_special_tokens=False):
        super().__init__()
        if no_special_tokens:
            self.symbols = []
            self.count = []
            self.indices = {}
            self.nspecial = 0

    @classmethod
    def load_list(cls, load_list):
        d = cls()
        for word in load_list:
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(1)
        return d

    def encode(self, line):
        line_encode = [self.index(word) for word in line]
        return line_encode


@register_task('task_suite_v2')
class TaskSuiteBase_v2(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--num_train', default=10000, type=int,
                            help='Num training examples')
        parser.add_argument('--num_test', default=10000, type=int,
                            help='Num test examples')
        parser.add_argument('--vocab_size', default=10, type=int,
                            help='Vocabulary size')
        parser.add_argument('--max_tasks', default=5, type=int,
                            help='Max tasks to precompute')
        parser.add_argument('--num_train_tasks', default=5, type=int,
                            help='Number of training tasks')
        parser.add_argument('--num_test_tasks', default=5, type=int,
                            help='Number of test tasks')
        parser.add_argument('--max_seq_len', default=16, type=int,
                            help='Maximum sequence length')
        parser.add_argument('--train_unseen_task', action='store_true',
                            help='Train on unseen task')
        parser.add_argument('--sample_num_tasks', default=1, type=int,
                            help='Num of tasks to sample for each iteration')
        parser.add_argument('--batch_version', action='store_true',
                            help='Batch update')
        parser.add_argument('--task_descriptions_dir', default='/tmp', type=str,
                            help='Location to write task descriptions')
        parser.add_argument('--eval_task_id', default=0, type=int,
                            help='Identifier of meta eval task')
        parser.add_argument('--load_tasks', default='/tmp', type=str,
                            help='Tasks file.')
        parser.add_argument('--no_training', action='store_true',
                            help='No fine-tuning.')
        parser.add_argument('--compositional', action='store_true',
                            help='Compositional task defn.')

    @classmethod
    def setup_task(cls, args, load_data=True, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.

        return TaskSuiteBase_v2(args, load_data)

    def __init__(self, args, load_data=True):
        super().__init__(args)
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.vocab_size = args.vocab_size
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.max_seq_len = args.max_seq_len
        self.train_unseen_task = args.train_unseen_task
        self.sample_num_tasks = args.sample_num_tasks
        self.batch_version = args.batch_version
        self.eval_task_id = args.eval_task_id
        self.no_training = args.no_training
        self.compositional = args.compositional
        self.num_classes = 4

        self.max_tasks = args.max_tasks
        assert self.num_train_tasks + self.num_test_tasks < self.max_tasks

        max_seq_len = self.max_seq_len
        vocab_size = self.vocab_size

        self.input_vocab = Dictionary_toy.load_list(range(vocab_size))

        self.cls_token = 'cls'
        self.cls_encode = self.input_vocab.add_symbol(self.cls_token)

        output_vocab = Dictionary_toy(no_special_tokens=True).load_list(range(max_seq_len))
        self.output_vocab = output_vocab

        self.label_map = {}
        self.label_encode = {}
        self.output_vocab_size = self.num_classes
        for i in range(self.num_classes):
            label_token = 'label%s' % i
            self.label_map[i] = label_token
            self.label_encode[i] = self.input_vocab.add_symbol(label_token)

        if load_data:
            task_generator = TaskGenerator(
                self.max_tasks,
                self.num_train,
                self.max_seq_len,
                self.vocab_size,
                self.num_classes,
                args.task_descriptions_dir)
            train_task_descriptions = task_generator.load_tasks(args.load_tasks + '/train.txt')
            # train_task_descriptions = task_generator.load_tasks(args.load_tasks + '/train_100.txt')

            self.train_task_descriptions = train_task_descriptions 

            if self.compositional:
                for task in self.train_task_descriptions:
                    for component in task.split('->'):
                        self.input_vocab.add_symbol(component)
                # primitives = [[], [], []]
                # min_max_inds = []
                # for task in self.train_task_descriptions:
                #     components = task.split('->')
                #     for i in range(3):
                #         primitives[i].append(components[i])
                # for i in range(3):
                #     indices = []
                #     for x in primitives[i]:
                #         indices.append(self.input_vocab.add_symbol(x))
                #     min_max_inds.append((np.min(indices), np.max(indices)))
                # self.task_embedding_inds = min_max_inds
    
            if self.train_unseen_task:
                test_task_descriptions = task_generator.load_tasks(args.load_tasks + '/test.txt')
    
                test_tasks = task_generator.generate_data(
                    test_task_descriptions, self.num_train, self.num_test, uniform_classes=True)
    
                train_examples = [task[0] for task in test_tasks]
                val_examples = [task[1] for task in test_tasks]
                test_examples = [task[2] for task in test_tasks]
    
                self.examples = {'train': train_examples, 'valid': val_examples, 'test': test_examples}
    
            else:
                test_task_descriptions = task_generator.load_tasks(args.load_tasks + '/val.txt')

                val_task_descriptions = test_task_descriptions[-self.num_test_tasks:]
                self.val_task_descriptions = val_task_descriptions 
    
                print('Generating data...')
                train_tasks = task_generator.generate_data(
                    train_task_descriptions, self.num_train, self.num_test)
                val_tasks = task_generator.generate_data(
                    val_task_descriptions, self.num_train, self.num_test)
                print('Done Generating data.')
    
                train_examples = [task[0] for task in train_tasks]
                val_examples = [task[0] for task in val_tasks]
    
                self.examples = {'train': train_examples, 'valid': val_examples}

            self.test_task_descriptions = test_task_descriptions 

    def construct_data(self, task_id, examples, task_description):

        sentences, lengths, labels = [], [], []

        task_description = task_description.split('->')

        for instance in examples:

            sentence, label = instance
            sentence = [task_id, self.cls_encode] + self.input_vocab.encode(sentence) 
            if self.compositional:
                sentence = sentence + self.input_vocab.encode(task_description)
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

    def construct_data_train(self, task_id, examples, task_description):
        return self.construct_data(task_id, examples, task_description)

    def construct_data_test(self, task_id, examples, train_examples, split, task_description):
        return self.construct_data(task_id, examples, task_description)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        input_vocab = self.input_vocab
        output_vocab = self.output_vocab

        examples = self.examples

        if self.train_unseen_task:

            task_id = self.eval_task_id
            # Note: Even though train and test task id's overlap, they index into different tables
            assert task_id < self.num_test_tasks

            task_descriptions = self.test_task_descriptions
            sentences, labels, lengths = self.construct_data_test(
                task_id, examples[split][task_id], examples['train'][task_id], split, task_descriptions[task_id])

            tgt_sizes = [label.shape[0] for label in labels]

            self.datasets[split] = LanguagePairDataset(
                src=sentences,
                src_sizes=lengths,
                src_dict=input_vocab,
                tgt=labels,
                tgt_sizes=tgt_sizes,
                tgt_dict=output_vocab,
                left_pad_source=False,
                max_target_positions=tgt_sizes[0],
                input_feeding=False,
            )
        else:
            dataset_map = OrderedDict()
            split_examples = examples[split]
            num_tasks = len(split_examples)

            if split == 'valid':
                task_descriptions = self.val_task_descriptions
            else:
                task_descriptions = self.train_task_descriptions

            for i in range(num_tasks):
                task_id = i
                sentences, labels, lengths = self.construct_data_train(
                    task_id, split_examples[i], task_descriptions[i])

                dataset_map[i] = LanguagePairDataset(
                    src=sentences,
                    src_sizes=lengths,
                    src_dict=input_vocab,
                    tgt=labels,
                    tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                    tgt_dict=output_vocab,
                    left_pad_source=False,
                    max_target_positions=1,
                    input_feeding=False,
                )
            self.datasets[split] = MultiCorpusSampledDataset(
                dataset_map, num_samples=self.sample_num_tasks)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab

    def _split(self, batch, size, k):
        return batch[k * size: (k + 1) * size]

    def _get_loss_tasks(self, sample, model, criterion):

        sample_sub = {}
        bs = sample['target'].shape[0]
        logs = []
        losses = []
        k = self.sample_num_tasks
        for i in range(k):
            sample_sub['target'] = self._split(sample['target'], bs // k, i)
            sample_sub['net_input'] = {}
            sample_sub['net_input']['src_tokens'] = self._split(sample['net_input']['src_tokens'], bs // k, i)
            sample_sub['net_input']['src_lengths'] = self._split(sample['net_input']['src_lengths'], bs // k, i)
            sample_sub['ntokens'] = sample['ntokens']

            loss, sample_size, logging_output = self._get_loss(sample_sub, model, criterion)
            losses.append(loss)
            logs.append(logging_output)

        loss = sum(losses) / k
        log_overall = {}
        for key in logging_output.keys():
            log_overall[key] = sum([log[key] for log in logs]) / k

        return loss, sample_size, log_overall

    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['split_data'] = split_data

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']

        # sample_size = sample['target'].size(0)
        sample_size = 1

        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

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
        if self.batch_version:
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        else:
            loss, sample_size, logging_output = self._get_loss_tasks(sample, model, criterion)
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        if model.training_mode == 'maml_meta':
            meta_grad = model.task_embeddings.weight.grad.sum(0)
            model.task_embedding_init.weight.grad.data.copy_(meta_grad)
            model.init_z_optimizer.step()

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        sample['net_input']['mode'] = 'eval'
        if 'meta' in model.training_mode:
            # Eval mode: Use 25% of the data to validation. The 75% is used for training by meta-learned
            # models and ignored by non-meta learning models.
            with torch.set_grad_enabled(True):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        for other_metrics in self.logging_diagnostics:
            agg_logging_outputs[other_metrics] = sum(
                log[other_metrics] for log in logging_outputs if other_metrics in log
            )
        return agg_logging_outputs

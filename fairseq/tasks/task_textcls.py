from collections import OrderedDict
from scipy import stats
import torch
import fairseq.tasks.loader as loader

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import FairseqTask, register_task

import numpy as np
from pdb import set_trace as bp


@register_task('task_textcls')
class Tasktextcls(FairseqTask):

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
        # data configuration
        parser.add_argument("--data_path", type=str,
                            default="/home/llajan/Distributional-Signatures/data",
                            help="path to dataset")
        parser.add_argument("--dataset", type=str, default="amazon",
                            help="name of the dataset. "
                            "Options: [20newsgroup, amazon, huffpost, "
                            "reuters, rcv1, fewrel]")
        parser.add_argument("--n_train_class", type=int, default=10,
                            help="number of meta-train classes")
        parser.add_argument("--n_val_class", type=int, default=5,
                            help="number of meta-val classes")
        parser.add_argument("--n_test_class", type=int, default=9,
                            help="number of meta-test classes")
    
        # model options
        parser.add_argument("--embedding", type=str, default="avg",
                            help=("document embedding method. Options: "
                                  "[avg, tfidf, meta, oracle, cnn]"))
        parser.add_argument("--classifier", type=str, default="nn",
                            help=("classifier. Options: [nn, proto, r2d2, mlp]"))
        parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                            help=("auxiliary embeddings (used for fewrel). "
                                  "Options: [pos, ent]"))
    
        # base word embedding
        parser.add_argument("--wv_path", type=str,
                            default="/mnt/brain6/scratch/llajan/",
                            help="path to word vector cache")
        parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                            help=("Name of pretrained word embeddings."))
        parser.add_argument("--finetune_ebd", action="store_true", default=False,
                            help=("Finetune embedding during meta-training"))
    
        parser.add_argument("--mode", type=str, default="train",
                            help=("Running mode."
                                  "Options: [train, test, finetune]"
                                  "[Default: test]"))
        parser.add_argument("--bert", default=False, action="store_true",
                            help=("set true if use bert embeddings "
                                  "(only available for sent-level datasets: "
                                  "huffpost, fewrel"))
        parser.add_argument("--meta_w_target", action="store_true", default=False,
                            help="use target importance score")


    @classmethod
    def setup_task(cls, args, load_data=True, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.

        return Tasktextcls(args, load_data)

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
        self.num_classes = 5

        self.max_tasks = args.max_tasks
        assert self.num_train_tasks + self.num_test_tasks < self.max_tasks

        max_seq_len = self.max_seq_len
        vocab_size = self.vocab_size

        self.vocab = Dictionary()

        self.label_map = {}
        self.label_encode = {}
        self.output_vocab_size = self.num_classes

        train_data, val_data, test_data, vocab = loader.load_dataset(args)
        if self.train_unseen_task:
            self.examples = {'train': test_data, 'valid': test_data, 'test': test_data}
            # self.examples = {'train': train_data, 'valid': val_data, 'test': test_data}
            # self.examples = {'train': val_data, 'valid': val_data, 'test': test_data}
        else:
            self.examples = {'train': train_data, 'valid': val_data, 'test': test_data}
        self.vocab_init = vocab.vectors
        self.vocab_size = vocab.vectors.size()[0]
        self.padding_idx = vocab.stoi['<pad>']
        self.cls_idx = vocab.stoi['[CLS]']

    def construct_data(self, task_id, examples):

        # sentences, lengths, labels = [], [], []

        text = examples['text']
        labels = examples['label']
        lengths = examples['text_len']

        # for i in range(len(text)):
        #     sentence = text[i][:lengths[i]]
        #     sentence = np.concatenate((np.array([self.cls_idx]), sentence))
        #     sentences.append(torch.LongTensor(sentence))
        #     lengths[i] += 1
        #     labels.append(torch.LongTensor([label[i]]))
        
        # sentences = [torch.LongTensor(sentence) for sentence in text]
        sentences = [torch.LongTensor(np.concatenate((np.array([self.cls_idx]), sentence))) for sentence in text]
        labels = [torch.LongTensor([label]) for label in labels]
        lengths = [length+1 for length in lengths]
        # lengths = [length+2 for length in lengths]

        return sentences, labels, lengths

    def construct_data_train(self, task_id, examples):
        return self.construct_data(task_id, examples)

    def construct_data_test(self, task_id, examples, train_examples, split):
        return self.construct_data(task_id, examples)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        examples = self.examples

        dataset_map = OrderedDict()
        split_examples = examples[split]
        num_tasks = len(split_examples)
        keys = list(split_examples.keys())

        for i in range(num_tasks):
            task_id = i
            sentences, labels, lengths = self.construct_data_train(
                task_id, split_examples[keys[i]])

            dataset_map[i] = LanguagePairDataset(
                src=sentences,
                src_sizes=lengths,
                src_dict=self.vocab,
                tgt=labels,
                tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                tgt_dict=self.vocab,
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
        return self.vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.vocab

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

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, epoch_itr=None):
        model.train()
        optimizer.zero_grad()
        sample['net_input']['num_tasks'] = self.sample_num_tasks
        sample['net_input']['optimizer'] = optimizer
        sample['net_input']['epoch_itr'] = epoch_itr
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)

        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        sample['net_input']['mode'] = 'eval'
        sample['net_input']['num_tasks'] = self.sample_num_tasks
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

import torch

from fairseq.tasks import register_task
from fairseq.tasks.task_suite_base_v2 import TaskSuiteBase_v2


@register_task('task_suite_snail')
class TaskSuite_concat(TaskSuiteBase_v2):

    @classmethod
    def setup_task(cls, args, load_data=True, **kwargs):
        return TaskSuite_concat(args, load_data)

    def __init__(self, args, load_data=True):
        super().__init__(args, load_data)

    def construct_data_train(self, task_id, examples, task_description):

        sentences, lengths, labels = [], [], []

        if self.compositional:
           task_description = task_description.split('->')
           task_description = self.input_vocab.encode(task_description) 

        for instance in examples:
            sentence, label = instance
            if self.compositional:
                sentence = task_description + self.input_vocab.encode(sentence) + [self.label_encode[label]]
            else:
                sentence = [task_id] + self.input_vocab.encode(sentence) + [self.label_encode[label]]
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

    def construct_data_test(self, task_id, examples, train_examples, split, task_description):

        sentences, lengths, labels = [], [], []

        train_sentences, _, _ = self.construct_data_train(task_id, train_examples, task_description)
        all_train_sentences = torch.cat(train_sentences, dim=0)

        if self.compositional:
           task_description = task_description.split('->')
           task_description = self.input_vocab.encode(task_description) 

        for instance in examples:

            sentence, label = instance
            if self.compositional:
               sentence = task_description + self.input_vocab.encode(sentence) + [self.label_encode[label]]
            else:
               sentence = [task_id] + self.input_vocab.encode(sentence) + [self.label_encode[label]]
            sentence = torch.LongTensor(sentence)
            sentence = torch.cat((all_train_sentences, sentence), dim=0)
            sentences.append(sentence)
            lengths.append(sentence.shape[0])
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

import sys

import torch
import torch.nn as nn
from torch.utils import data as tds

# linear elman rnn, 
class ControllableElmanGenerator():
    def __init__(self, vocab_size, hdim,
                 read_inputs=True,
                 out_mult = 5,
                 random_init=False):
        self.hdim = hdim
        outmap = torch.randn(vocab_size, hdim)
        outmap = outmap/outmap.norm(2,1).unsqueeze(1)
        self.outmap = out_mult*outmap
        self.input_embedding = torch.randn(vocab_size, hdim)
        self._A = torch.randn(hdim, hdim, hdim)
        self.register_task(torch.ones(hdim))
        self.tanh = nn.Tanh()
        self.read_inputs = read_inputs
        self.random_init = random_init
        
    def register_task(self, x, init_h=None):
        if not init_h:
            self.init_h = torch.ones(self.hdim)
        A = torch.zeros(self.hdim, self.hdim)
        for i in range(self.hdim):
            A[i,:] = self._A[i,:,:]@x
        u, s, v = torch.svd(A)
        self.A = self.A = u@v.t()
                           
    def generate(self, n):
        h = self.init_h
        generated = []
        if self.random_init:
            x = torch.randn(self.hdim)
        else:
            x = torch.zeros(self.hdim)
        for i in range(n):
            h = self.A@h
            if self.read_inputs:
                h = h + x
            h = self.tanh(h)
            C = torch.distributions.categorical.Categorical(logits=self.outmap@h)
            generated.append(C.sample().item())
            x = self.input_embedding[generated[-1]]
        return generated

    
if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_tasks", type=int, default=2000)
    parser.add_argument("--num_test_tasks", type=int, default=100)
    parser.add_argument("--num_train_examples", type=int, default=500)
    parser.add_argument("--num_test_examples", type=int, default=500)
    parser.add_argument("--out_mult", type=int, default=4)
    parser.add_argument("--hdim", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--intrinsic_dim", type=int, default=1)
    parser.add_argument("--generation_length", type=int, default=10)
    parser.add_argument("--save_path", default='/checkpoint/aszlam/laja/rnn_tasks.pkl')
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--read_inputs", action='store_true')
    args = parser.parse_args()

    
    generator = ControllableElmanGenerator(args.vocab_size,
                                           args.hdim,
                                           read_inputs=args.read_inputs,
                                           out_mult=args.out_mult,
                                           random_init=args.random_init)
    ntasks = args.num_train_tasks + 2 * args.num_test_tasks
    tasks = torch.randn(ntasks, args.intrinsic_dim + 1)
    tasks = tasks/tasks.norm(2,1).unsqueeze(1)
    hmap = torch.randn(args.hdim, args.intrinsic_dim + 1)
    l = args.generation_length//2
    out = []
    for i in tqdm(range(ntasks)):
        task_out = []
        generator.register_task(hmap@tasks[i])
        train_strings = set()
        valid_strings = set()
        num_failures = 0
        examples_per_task = args.num_train_examples + args.num_test_examples * 2
        for j in range(examples_per_task):
            if j < args.num_train_examples:
                set_partition = 'train'
            elif j < (args.num_train_examples + args.num_test_examples):
                set_partition = 'valid'
            else:
                set_partition = 'test'
            x = generator.generate(args.generation_length)
            x_str = ' '.join(map(str, x))
            while (set_partition == 'valid' and x_str in train_strings) or (set_partition == 'test' and (x_str in train_strings or x_str in valid_strings)):
                num_failures += 1
                if num_failures >= 10000:
                    sys.exit('Cannot generate enough samples. Already have %d samples.' % j)
                x = generator.generate(args.generation_length)
                x_str = ' '.join(map(str, x))

            if set_partition == 'train':
                train_strings.add(x_str)
            elif set_partition == 'valid':
                valid_strings.add(x_str)

            task_out.append((x[:l], x[l:]))
        out.append(task_out)
    
    f = open(args.save_path, 'wb')
    data = {'num_train_tasks': args.num_train_tasks,
            'num_test_tasks': args.num_test_tasks,
            'data': out}
    pickle.dump(data, f)
    f.close()

import torch
import torch.nn as nn
from torch.utils import data as tds

# linear elman rnn, 
class ControllableElmanGenerator():
    def __init__(self, vocab_size, hdim,
                 read_inputs=True,
                 out_mult = 5):
        self.hdim = hdim
        outmap = torch.randn(vocab_size, hdim)
        outmap = outmap/outmap.norm(2,1).unsqueeze(1)
        self.outmap = out_mult*outmap
        self.input_embedding = torch.randn(vocab_size, hdim)
        self._A = torch.randn(hdim, hdim, hdim)
        self.register_task(torch.ones(hdim))
        self.tanh = nn.Tanh()
        self.read_inputs = read_inputs
        
    def register_task(self, x, init_h=None):
        if not init_h:
            self.init_h = torch.ones(self.hdim)
        A = torch.zeros(self.hdim, self.hdim)
        for i in range(self.hdim):
            A[i,:] = self._A[i,:,:]@x
        u, s, v = torch.svd(A)
        self.A = u
                           
    def generate(self, n):
        h = self.init_h
        generated = []
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
    parser.add_argument("--examples_per_task", type=int, default=1500)
    parser.add_argument("--out_mult", type=int, default=4)
    parser.add_argument("--hdim", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--intrinsic_dim", type=int, default=1)
    parser.add_argument("--generation_length", type=int, default=10)
    parser.add_argument("--save_path", default='/checkpoint/aszlam/laja/rnn_tasks.pkl')
    args = parser.parse_args()

    
    generator = ControllableElmanGenerator(args.vocab_size,
                                           args.hdim,
                                           read_inputs=True,
                                           out_mult=args.out_mult)
    ntasks = args.num_train_tasks + args.num_test_tasks
    tasks = torch.randn(ntasks, args.intrinsic_dim + 1)
    tasks = tasks/tasks.norm(2,1).unsqueeze(1)
    hmap = torch.randn(args.hdim, args.intrinsic_dim + 1)
    l = args.generation_length//2
    out = []
    for i in tqdm(range(ntasks)):
        task_out = []
        generator.register_task(hmap@tasks[i])
        for j in range(args.examples_per_task):
            x = generator.generate(args.generation_length)
            task_out.append((x[:l], x[l:]))
        out.append(task_out)
    
    f = open(args.save_path, 'wb')
    data = {'num_train_tasks': args.num_train_tasks,
            'num_test_tasks': args.num_test_tasks,
            'data': out}
    pickle.dump(data, f)
    f.close()

import numpy as np
from scipy import stats
import itertools
import random
import ast

def make_apply_map(m):
    def apply_map(x):
        t = m.get(x)
        if t is not None:
            return t
        else:
            return x
    return apply_map

###################
# index transforms
###################

class Replace():
    #transform tokens via an index map
    def __init__(self, tform_data):
        self.tform = make_apply_map(tform_data['index_tform'])
        #tform is a dict of {source:target, ..., }
    def transform(self, x):
        return [self.tform(e) for e in x]

# permutation
# cycle

################################
# fixed size spatial transforms
################################

class SpatialPermute():
    #transform tokens via a permutation of the locations
    def __init__(self, tform_data):
        self.tform = tform_data['spatial_tform']
    def transform(self, x):
        #tform is a dict of {source:target, ..., }
        z = x.copy()
        for s, t in self.tform.items(): 
            z[t] = x[s]
        return z


################################
# size change spatial transforms
################################
def selectSeq(x, idx, reverse=False):
    idx = sorted(idx)
    if not reverse:
        return [x[i] for i in idx]
    else:
        m = [0 if i in idx else 1 for i in range(len(x))]
        return list(itertools.compress(x, m))


def insertSeq(x, seq, idx):
    idx = sorted(idx) 
    z = x.copy()
    for i, bias in zip(idx, range(len(idx))):
        z.insert(i+bias, seq[bias])
    return z


def replaceSeq(x, val, idx):
    return [val if i in idx else x[i] for i in range(len(x))]


class DeleteMask():
    #delete tokens in the mask
    def __init__(self, tform_data):
        self.mask = tform_data['mask']
    def transform(self, x):
        #mask is a list of indices
        return selectSeq(x, self.mask, reverse=True)


class InsertMask():
    def __init__(self, tform_data):
        #mask is a list of indices
        #inserts each entry in Tensor seq
        #to the left of the corresponding entry in mask
        self.mask = tform_data['mask']
        self.seq = tform_data['seq']
    def transform(self, x):
        return insertSeq(x, self.seq, self.mask)


class InsertCopyMask():
    def __init__(self, tform_data):
        #mask is a list of indices
        #copies each entry at location seq one to the left, 
        # e.g. if x = 1,2,3 and mask=[1,2] returns 1,2,2,3,3
        self.mask = tform_data['mask']
    def transform(self, x):
        return insertSeq(x, selectSeq(x, self.mask), self.mask)


################################
# Mixed space and index
################################

class OverwriteMask():
    #fill the mask with a target index
    def __init__(self, tform_data):
        self.targetid = tform_data['targetid']
        self.mask = tform_data['mask']
    def transform(self, x):
        return replaceSeq(x, self.targetid, self.mask)


class DeleteByIndex():
    # delete all tokens with the given value
    def __init__(self, tform_data):
        self.targetids = tform_data['targetids']
    def transform(self, x):
        m = [0 if e in self.targetids else 1 for e in x]
        return list(itertools.compress(x, m))


class MinMaxMaskFill():
    #fill a mask by the min or max in that spatial mask
    def __init__(self, tform_data):
        self.mask = tform_data['mask']
        self.maxminf = max if tform_data['maxmin'] == 'max' else min
    def transform(self, x):
        mval = self.maxminf(selectSeq(x, self.mask))
        return replaceSeq(x, mval, self.mask)


def make_random_cycle(N, cycle_length):
    perm = list(range(N))
    random.shuffle(perm)
    perm = perm[:cycle_length]
    out = {perm[-1]:perm[0]}
    for i in range(cycle_length - 1):
        out[perm[i]] = perm[i+1]
    return out


class TaskGeneratorV2():

    def __init__(self, max_tasks, seqlen, vocab_size, subseq_len, logdir=None):

        self.vocab_size = vocab_size
        self.seqlen = seqlen
        self.max_tasks = max_tasks

        self.subseq_len = subseq_len
        assert self.subseq_len < self.seqlen and self.subseq_len > 0

        self.logdir = logdir

        self.fn_map = self.function_mapping()


    def generate_tasks(self):

        tasks_explored = set()

        all_inds = list(range(self.seqlen))
        all_words = list(range(self.vocab_size))
        transforms = list(self.fn_map.keys())

        tasks_iter = 0
        while tasks_iter < self.max_tasks:
            print(tasks_iter)

            subseq_len = random.randint(1, self.subseq_len)

            #just generate everything, most won't be used by any particular task
            tform_data = {'mask': random.sample(all_inds, subseq_len),
                          'maxmin': random.choice(['max', 'min']),
                          'targetid': random.randint(0, self.vocab_size - 1),
                          'targetids': random.sample(all_words, subseq_len),
                          'seq': [random.randint(0, self.vocab_size - 1)] * subseq_len,
                          'spatial_tform': make_random_cycle(self.seqlen, subseq_len),
                          'index_tform': make_random_cycle(self.vocab_size, subseq_len)}

            task_description = self.task_description(random.choice(transforms), tform_data)
            
            if task_description in tasks_explored:
                continue           

            tasks_explored.add(task_description)
            tasks_iter += 1

        if self.logdir is not None:
            with open(self.logdir, 'w') as f:
                f.write('\n'.join(tasks_explored))

        return list(tasks_explored)


    def task_description(self, transform, tform_data):
        return transform + ' -> ' + str(tform_data)


    def parse_task(self, task_description):
        words = task_description.split(' -> ')
        assert len(words) == 2

        task = words[0]
        tform_data = ast.literal_eval(words[1])
        
        return self.fn_map[task](tform_data)


    def load_tasks(self, tasks_file):
        with open(tasks_file, 'r') as f:
            lines = f.readlines()
            tasks = [line.strip() for line in lines]
        return tasks


    def generate_data(self, tasks, num_train, num_test):

        rng = np.random.RandomState(seed=1234)

        all_data = []

        num_examples = num_train + 2 * num_test

        for task in tasks:
            task = self.parse_task(task)
            data = []
            for n in range(num_examples):
                source = list(rng.randint(self.vocab_size, size=(self.seqlen,)))
                target = task.transform(source)

                data.append((source, target))

            assert len(data) == num_examples

            train = data[:num_train]
            val = data[num_train:num_train+num_test]
            test = data[-num_test:]

            rng.shuffle(train)

            all_data.append((train, val, test))

        return all_data


    def function_mapping(self):
        fn_map = {
            'Replace': Replace,
            'SpatialPermute': SpatialPermute,
            'DeleteMask': DeleteMask,
            'InsertMask': InsertMask,
            'InsertCopyMask': InsertCopyMask,
            'OverwriteMask': OverwriteMask,
            'DeleteByIndex': DeleteByIndex,
            'MinMaxMaskFill': MinMaxMaskFill
        }

        return fn_map


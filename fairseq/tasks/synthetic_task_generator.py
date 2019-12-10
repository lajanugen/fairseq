import numpy as np
from scipy import stats
import itertools


class TaskGenerator():

    def __init__(self, max_tasks, seqlen, vocab_size, logdir=None):

        self.vocab_size = vocab_size
        self.seqlen = seqlen
        self.max_tasks = max_tasks

        self.logdir = logdir

        self.construct_tasks()
        self.fn_map = self.function_mapping()


    def construct_tasks(self):

        transforms = []
        muls = []
        adds = []
        for k in [2, 3, 5, 7, 9, 10, 11]:
            muls.append(('mul', k))
        transforms.extend(muls)
        for k in range(self.vocab_size):
            adds.append(('add', k))
        transforms.extend(adds)
        for k in [2, 3]:
            transforms.append(('div', k))
        for k in [5, 6, 7, 8, 9, 10, 11]:
            transforms.append(('mod', k))

#        for k in range(1, self.vocab_size + 1):
#            muls.append(('mul', k))
#            adds.append(('add', k))
#            transforms.append(('div', k))
#            transforms.append(('mod', k))
#        transforms.extend(muls)
#        transforms.extend(adds)

        subseq = []

        for k, n in itertools.product(range(self.vocab_size // 2), range(self.vocab_size)):
            if k != n:
                subseq.append(('replace-k', k, n))

        for i in range(self.seqlen):
            base = ('replace-ith', i)
            for m in muls:
                subseq.append(base + m)
            for a in adds:
                subseq.append(base + a)
            for m, a in itertools.product(muls, adds):
                subseq.append(base + m + a)

        for i, j in itertools.product(range(self.seqlen), range(self.seqlen)):
            if i < j:
                subseq.append(('replace-ith-jth', i, j))

        for i in range(self.seqlen):
            subseq.append(('replace-ith-next', i, 'sum'))
            subseq.append(('replace-ith-next', i, 'diff'))

        reorder = [
            ('sort-gt', ),
            ('sort-lt', ),
            ('reverse', )
        ]

        for i, j in itertools.product(range(self.seqlen), range(self.seqlen)):
            if i < j:
                reorder.append(('swap', i, j))

        for i in range(self.seqlen - 1):
            reorder.append(('shift', i))

        self.transforms = transforms
        self.subseq = subseq
        self.reorder = reorder

        self.transforms_taskid = {}
        self.subseq_taskid = {}
        self.reorder_taskid = {}

        for i in range(len(transforms)):
            self.transforms_taskid[' '.join(map(str, transforms[i]))] = i
        for i in range(len(subseq)):
            self.subseq_taskid[' '.join(map(str, subseq[i]))] = i + len(transforms)
        for i in range(len(reorder)):
            self.reorder_taskid[' '.join(map(str, reorder[i]))] = i + len(transforms) + len(subseq)
        

    def generate_tasks(self):

        tasks_explored = set()

        rng = np.random.RandomState(seed=1234)

        tasks_iter = 0
        while True:
            print(tasks_iter)
            transform = self.transforms[rng.choice(len(self.transforms))]
            subseq = self.subseq[rng.choice(len(self.subseq))]
            reorder = self.reorder[rng.choice(len(self.reorder))]
            task_description = self.task_description(transform, subseq, reorder)
            
            if task_description in tasks_explored:
                continue           

            data = self.generate_data_single_task(task_description, 100, rng)  # test that the task makes sense
            if len(data) == 0:
                print('bad task: %s' % task_description)
                continue

            tasks_explored.add(task_description)
           
            tasks_iter += 1
            if tasks_iter >= self.max_tasks:
                break

        if self.logdir is not None:
            with open(self.logdir, 'w') as f:
                f.write('\n'.join(tasks_explored))

        return list(tasks_explored)


    def generate_all_tasks(self):
        rng = np.random.RandomState(seed=1234)

        tasks_iter = 0
        tasks_explored = []
        for transform, subseq, reorder in itertools.product(self.transforms, self.subseq, self.reorder):
            print(tasks_iter)
            tasks_iter += 1

            task_description = self.task_description(transform, subseq, reorder)
            
            data = self.generate_data_single_task(task_description, 100, rng)  # test that the task makes sense
            if len(data) == 0:
                print('bad task: %s' % task_description)
                continue

            tasks_explored.append(task_description)

        if self.logdir is not None:
            with open(self.logdir, 'w') as f:
                f.write('\n'.join(tasks_explored))

        return tasks_explored


    def task_description(self, transform, subseq, reorder):
        transform = map(str, transform)
        subseq = map(str, subseq)
        reorder = map(str, reorder)

        return ' -> '.join([' '.join(transform), ' '.join(subseq), ' '.join(reorder)])


    def get_task_ids(self, task_description):
        subtasks = task_description.split(' -> ')

        assert subtasks[0] in self.transforms_taskid
        assert subtasks[1] in self.subseq_taskid
        assert subtasks[2] in self.reorder_taskid

        return (self.transforms_taskid[subtasks[0]], self.subseq_taskid[subtasks[1]], self.reorder_taskid[subtasks[2]])


    def load_tasks(self, tasks_file):
        with open(tasks_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines


    def generate_data(self, tasks, num_train, num_test):

        rng = np.random.RandomState(seed=1234)

        all_data = []

        num_examples = num_train + 2 * num_test

        for task in tasks:
            data = self.generate_data_single_task(task, num_examples, rng, allow_fail=False)
         
            assert len(data) == num_examples

            train = data[:num_train]
            val = data[num_train:num_train+num_test]
            test = data[-num_test:]

            rng.shuffle(train)

            all_data.append((train, val, test))

        return all_data


    def parse_task(self, task):
        transform, subseq, reorder = task.split(' -> ')
        transform = transform.split()
        transform[1] = int(transform[1])

        subseq = subseq.split()
        subseq[1] = int(subseq[1])
        if len(subseq) >= 4:  # replace-ith
            for i in range(3, len(subseq), 2):
                subseq[i] = int(subseq[i])
            for i in range(2, len(subseq), 2):
                subseq[i] = self.fn_map[subseq[i]]
        elif subseq[0] != 'replace-ith-next':
            subseq[2] = int(subseq[2])

        reorder = reorder.split()
        if len(reorder) > 1:
            for i in range(1, len(reorder)):
                reorder[i] = int(reorder[i])

        return transform, subseq, reorder


    def generate_output(self, transform, subseq, reorder, in_seq):
        x_tf = self.fn_map[transform[0]](in_seq, transform[1])
        if len(x_tf) != len(in_seq):
            return []

        x_subseq = self.fn_map[subseq[0]](x_tf, subseq[1:])
        if len(x_subseq) != len(in_seq):
            return []

        x_final = self.fn_map[reorder[0]](x_subseq, reorder)

        return x_final


    def generate_data_single_task(self, task, num_examples, rng, allow_fail=True):
        transform, subseq, reorder = self.parse_task(task)

        input_strings = set()
        data = []
        fail_count = 0
        while True:
            if allow_fail and fail_count >= 1000:
                return []

            x = list(rng.randint(self.vocab_size, size=(self.seqlen,)))

            x_str = ' '.join(map(str, x))
            if x_str in input_strings:
                continue
            input_strings.add(x_str)

            x_final = self.generate_output(transform, subseq, reorder, x)

            if len(x_final) != len(x):
                fail_count += 1
                continue

            data.append((x, x_final))

            if len(data) == num_examples:
                break

        return data


    def function_mapping(self):

        def mul(x, k):
            return [(k * e) % self.vocab_size for e in x]

        def add(x, k):
            return [(k + e) % self.vocab_size for e in x]

        def div(x, k):
            return [e // k for e in x]

        def mod(x, k):
            return [e % k for e in x]

        def replacek(x, args):
            orig = args[0]
            target = args[1]
            if not orig in x:
                return []
            return [target if e == orig else e for e in x]

        def replaceith(x, args):
            k = [x[args[0]]]
            for f, n in zip(args[1::2], args[2::2]):
                k = f(k, n)
            x[args[0]] = k[0]
            return x
     
        def replaceij(x, args):
            x[args[0]] = x[args[1]]
            return x

        def replaceinext(x, args):
            id = args[0]
            id_next = (id + 1) % len(x)
            if args[1] == 'sum':
                x[id] = (x[id] + x[id_next]) % self.vocab_size
            elif args[1] == 'diff':
                x[id] = abs(x[id] - x[id_next])

            return x

        def sortx(x, args):
            x.sort(reverse=(args[0] == 'sort-gt'))
            return x

        def swapx(x, args):
            x[args[1]], x[args[2]] = x[args[2]], x[args[1]]
            return x

        def shiftk(x, args):
            return [x[(i + args[1]) % len(x)] for i in range(len(x))]

        def reverse(x, args):
            x.reverse()
            return x

        fn_map = {
            'mul': mul,
            'add': add,
            'div': div,
            'mod': mod,
            'replace-k': replacek,
            'replace-ith': replaceith,
            'replace-ith-jth': replaceij,
            'replace-ith-next': replaceinext,
            'sort-gt': sortx,
            'sort-lt': sortx,
            'swap': swapx,
            'shift': shiftk,
            'reverse': reverse
        }

        return fn_map


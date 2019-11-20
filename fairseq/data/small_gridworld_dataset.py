from collections import defaultdict
from heapq import *
import torch

def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return -1, None

def raster(h,w,sl):
    return h*sl + w

def deraster(x, sl):
    w = x%sl
    h = (x - w)//sl
    return h, w

def flatten(nest_path, path):
    if len(nest_path) == 0:
        return path
    else:
        path.append(nest_path[0])
        return flatten(nest_path[1], path)
    
class GW():
    def __init__(self,
                 sidelength=10,
                 num_blobs=6,
                 blob_size=2,
                 gen_len = 12):
        self.sidelength = sidelength
        self.num_blobs = num_blobs
        self.blob_size = blob_size
        self.gen_len = gen_len

    def register_task(self, start_h, start_w, end_h, end_w):
        assert(start_h>=0)
        assert(start_h<self.sidelength)
        assert(start_w>=0)
        assert(start_w<self.sidelength)
        self.start_h = start_h
        self.end_h = end_h
        self.start_w = start_w
        self.end_w = end_w

    def check_reachable(self, h, w, blob_pos):
        if h < 0 or w < 0:
            return False
        if h>= self.sidelength or w >= self.sidelength:
            return False
        for i in range(len(blob_pos)//2):
            bh = blob_pos[2*i]
            bw = blob_pos[2*i + 1]
            if h >= bh and h < bh+self.blob_size and w >= bw and w < bw+self.blob_size:
                return False
        return True
        
    def generate_world_and_path(self):
        ok = False
        count = 100
        while not ok:
            count += 1
            blob_pos = torch.randint(0, self.sidelength, (2*self.num_blobs,)).tolist()
            if self.check_reachable(self.start_h, self.start_w, blob_pos):
                if self.check_reachable(self.end_h, self.end_w, blob_pos):
                    ok = True
            if count > 1000:
                raise Exception('world is too small?  cant find place for blobs')
        self.blob_pos = blob_pos
        edges = []
        #(h,w)--> h*self.sidelength + w
        for h in range(self.sidelength):
            for w in range(self.sidelength):
                for u in [h + 1, h, h - 1]:
                    for v in [w + 1, w, w - 1]:
                        if self.check_reachable(u, v, blob_pos):
                            source = raster(h, w, self.sidelength)
                            target = raster(u, v, self.sidelength)
                            edges.append((source, target, 1))
#                for u in [h + 1, h - 1]:
#                    if self.check_reachable(u, w, blob_pos):
#                        source = raster(h, w, self.sidelength)
#                        target = raster(u, w, self.sidelength)
#                        edges.append((source, target, 1))
#                for v in [w + 1, w - 1]:
#                    if self.check_reachable(h, v, blob_pos):
#                        source = raster(h, w, self.sidelength)
#                        target = raster(h, v, self.sidelength)
#                        edges.append((source, target, 1))
                        
        source = raster(self.start_h, self.start_w, self.sidelength)
        target = raster(self.end_h, self.end_w, self.sidelength)
        cost, path = dijkstra(edges, source, target)
        self.edges = edges
        return path
    
    def generate(self):
        path = self.generate_world_and_path()
        source = []
        for i in range(0,len(self.blob_pos), 2):
            source.append(raster(self.blob_pos[i],
                                 self.blob_pos[i+1],
                                 self.sidelength))
        
        if path is None:
            return (source, [])
        else:
            fpath = flatten(path,[])[::-1]
            targetlen = min(len(fpath), self.gen_len)
            target = [raster(self.end_h, self.end_w, self.sidelength) + self.sidelength**2]*targetlen
            for i in range(targetlen):
                target[i] = fpath[i] + self.sidelength**2
            return (source, target)
            
            

    def display_ascii(self):
        m = ''
        for h in range(self.sidelength):
#            for w in range(self.sidelength):
#                m = m + '   '
#            m = m + '\n'
            for w in range(self.sidelength):
                if h==self.start_h and w==self.start_w:
                    m = m + '  S  '
                elif h==self.end_h and w==self.end_w:
                    m = m + '  E  '
                elif self.check_reachable(h, w, self.blob_pos):
                    m = m + '  .  '
                else:
                    m = m + '  B  '
            m = m + '\n'
            for w in range(self.sidelength):
                m = m + '     '
            m = m + '\n'
        return m

if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--just_tlist", action="store_true")
    parser.add_argument("--tlist_path", default='')
    parser.add_argument("--sidelength", type=int, default=10)
    parser.add_argument("--num_blobs", type=int, default=8)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--start_point", type=int, default=-1)
    parser.add_argument("--num_test_tasks", type=int, default=100)
    parser.add_argument("--examples_per_task", type=int, default=1500)
    parser.add_argument("--gen_len", type=int, default=12)
    parser.add_argument("--save_path", default='/checkpoint/aszlam/laja/gridworld_tasks.pkl')
    args = parser.parse_args()

    assert(args.start_point < args.sidelength**2)
    if args.start_point >= 0:
        ntasks = args.sidelength**2
        S = args.start_point
    else:
        ntasks = args.sidelength**4

    if args.just_tlist:
        tlist = torch.randperm(ntasks).tolist()
        f = open(args.tlist_path, 'wb')
        pickle.dump(tlist, f)
        f.close()
        quit()
    else:
        if args.tlist_path == '':
            tlist = torch.randperm(ntasks).tolist()
        else:
            f = open(args.tlist_path, 'rb')
            tlist = pickle.load(f)
      
    generator = GW(sidelength=args.sidelength, num_blobs=args.num_blobs, gen_len=args.gen_len)
    out = []
    if args.end_id == -1:
        end_id = ntasks
    else:
        end_id = args.end_id
        
    assert(end_id <= ntasks)
    
    for i in tqdm(range(args.start_id, end_id)):
        taskid = tlist[i]
        if args.start_point >= 0:
            T = taskid
        else:
            S, T = deraster(taskid, args.sidelength**2)
        start_h, start_w = deraster(S, args.sidelength)
        end_h, end_w = deraster(T, args.sidelength)
        generator.register_task(start_h, start_w, end_h, end_w)
        task_out = []
        for j in range(args.examples_per_task):
            x = generator.generate()
            while len(x[1]) == 0:
                x = generator.generate()
            task_out.append(x)
        out.append(task_out)

    if args.end_id > 0 :
        import os
        bn = os.path.basename(args.save_path)
        dn = os.path.dirname(args.save_path)
        sp = 'chunk_' + str(end_id) + '_' + bn
        sp = os.path.join(dn, sp)
    else:
        sp = args.save_path        
    f = open(sp, 'wb')
    data = {'num_train_tasks': args.sidelength**2 - args.num_test_tasks,
            'num_test_tasks': args.num_test_tasks,
            'task_permute': tlist,
            'data': out}
    pickle.dump(data, f)
    f.close()

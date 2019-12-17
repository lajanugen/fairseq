from collections import defaultdict
from heapq import *
import random
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

def build_all_waypoints(args):
    train_waypoints = []
    test_waypoints = []
    for s in range(args.num_waypoints):
        train_waypoints.append([])
        test_waypoints.append([])
        for i in range(args.sidelength):
            for j in range(args.sidelength):
                if torch.rand(1).item() < args.pct_unseen_components:
                    test_waypoints[s].append((i,j))
                else:
                    train_waypoints[s].append((i,j))
    return train_waypoints, test_waypoints

    
class GW():
    def __init__(self,
                 sidelength=10,
                 num_blobs=6,
                 blob_size=2,
                 gen_len = 15):
        self.sidelength = sidelength
        self.num_blobs = num_blobs
        self.blob_size = blob_size
        self.gen_len = gen_len

    def hash_waypoints(self, waypoints):
        return raster(self.waypoints[-1][0], self.waypoints[-1][1], self.sidelength) + self.sidelength**2

    def register_task(self, waypoints):
        self.waypoints = waypoints
        self.waypoints_idx = [raster(w[0], w[1], self.sidelength) + self.sidelength**2  for w in waypoints]
        self.waypoints_dict = {}
        for i in range(len(self.waypoints)):
            self.waypoints_dict[self.waypoints[i]] = i

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
        count = 0
        while not ok:
            count += 1
            blob_pos = torch.randint(0, self.sidelength, (2*self.num_blobs,)).tolist()
            if all([self.check_reachable(h,w, blob_pos) for (h,w) in self.waypoints]):
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
        path = []
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i + 1]
            start = raster(start[0], start[1], self.sidelength)
            end = raster(end[0], end[1], self.sidelength)
            cost, local_path = dijkstra(edges, start, end)
            if local_path:
                fpath = flatten(local_path,[])[::-1]
            else:
                fpath = []
            for p in fpath[:-1]:
                path.append(p )
        path.append(end)
        self.edges = edges
        self.path = path
    
    def generate(self):
        self.generate_world_and_path()
        source = []
        for i in range(0,len(self.blob_pos), 2):
            source.append(raster(self.blob_pos[i],
                                 self.blob_pos[i+1],
                                 self.sidelength))

        if self.path is None:
            return (source, [self.waypoints_idx[0]]*self.gen_len)
        else:
            target = [self.waypoints_idx[-1]]*self.gen_len
            for i in range(min(len(self.path), self.gen_len)):
                target[i] = self.path[i] + self.sidelength**2
            return (source, target)
            
            

    def display_ascii(self):
        m = ''
        for h in range(self.sidelength):
#            for w in range(self.sidelength):
#                m = m + '   '
#            m = m + '\n'
            for w in range(self.sidelength):
                wid = self.waypoints_dict.get((h,w))
                if wid is not None:
                    m = m + '  ' + str(wid) + '  '
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
    parser.add_argument("--just_waypoints_list", action="store_true")
    parser.add_argument("--waypoints_list_path", default='')
    parser.add_argument("--sidelength", type=int, default=10)
    parser.add_argument("--num_blobs", type=int, default=8)
    parser.add_argument("--num_train_tasks", type=int, default=10000)
    parser.add_argument("--num_test_tasks", type=int, default=100)
    parser.add_argument("--num_waypoints", type=int, default=3)
    parser.add_argument("--pct_unseen_components", type=float, default=.1)
    parser.add_argument("--examples_per_task", type=int, default=1500)
    parser.add_argument("--save_path", default='/checkpoint/aszlam/laja/gridworld_tasks_comp.pkl')
    args = parser.parse_args()

      
    generator = GW(sidelength=args.sidelength, num_blobs=args.num_blobs)
    out = []

    if args.just_waypoints_list:
        train_waypoints, test_waypoints = build_all_waypoints(args)
        f = open(args.waypoints_list_path, 'wb')
        pickle.dump([train_waypoints, test_waypoints], f)
        f.close()
        quit()
    else:
        if args.waypoints_list_path == '':
            train_waypoints, test_waypoints = build_all_waypoints(args)
        else:
            f = open(args.waypoints_list_path, 'rb')
            l = pickle.load(f)
            train_waypoints, test_waypoints = l
            f.close()
    
    
    for i in tqdm(range(args.num_train_tasks)):
        waypoints = []
        for s in range(args.num_waypoints):
            waypoints.append(random.choice(train_waypoints[s]))
        generator.register_task(waypoints)
        task_out = []
        for j in range(args.examples_per_task):
            x = generator.generate()
            task_out.append(x)
        out.append((task_out, generator.waypoints_idx, None))

    for i in tqdm(range(args.num_test_tasks)):
        waypoints = []
        new_task_position = random.choice([0,1,2])
        for s in range(args.num_waypoints):
            if s == new_task_position:
                waypoints.append(random.choice(test_waypoints[s]))
            else:
                waypoints.append(random.choice(train_waypoints[s]))
        generator.register_task(waypoints)
        task_out = []
        for j in range(args.examples_per_task):
            x = generator.generate()
            task_out.append(x)
        out.append((task_out, generator.waypoints_idx, new_task_position))
        


    sp = args.save_path        
    f = open(sp, 'wb')
    pickle.dump(out, f)
    f.close()


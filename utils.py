import numpy as np

def get_dict_item(d):
    l = []; n = []
    if type(d) is dict:
        keys, items = zip(*sorted(d.items(), key = lambda x:x[0]))
        l += list(keys)
        for key, item in zip(keys, items):
            _l, _n = get_dict_item(item)
            l += [key + '/' +i for i in _l]
            n += [_n]
    else:
        return d, 0
    return l, n

def get_idxs(flat_tree_form):
    '''
    return parents, children, childless
      parents: list of lists of idxs of parents of each child
      children: list of lists of idxs of children of each parent
      childless: list of idxs of the childless
    '''
    parents = []
    children = [[] for _ in range(len(flat_tree_form) - 1)]
    childless = []
    mp = []
    p = []
    c = 0
    for n in flat_tree_form:
        for i in range(c, c + n): parents += [p + [i]]
        if len(p) > 0: children[p[-1]] = list(range(c, c+n))
        if n == 0:
            childless += [p[-1]]
            p[-1] += 1
            while p[-1] == mp[-1] and len(p) > 1:
                p.pop(-1); mp.pop(-1)
                p[-1] += 1
        else:
            p += [c]
            mp += [c + n]
        c += n
    return parents, children, childless

def flatten(l):
    for i in l:
        if type(i) is list:
            yield len(i)
            for f in flatten(i):
                yield f
        else:
            yield i

def convert_to_child(parent, children):
    while len(children[parent]) > 0:
        parent =  np.random.choice(children[parent])
    return parent

def gen_ex(exposed_y, parents, children, noise_std = .1):
    '''
    Toy data generation function
    '''
    true_y = np.array([convert_to_child(i, children) for i in exposed_y])
    true_x = np.stack([sum([k%4 * (1/4) ** j for j, k in enumerate(parents[i])]) for i in true_y])[:, None]
    exposed_x = np.random.normal((np.repeat(true_x, 16, 1) * 10) % np.linspace(0.5, 1, 16)[None, :], noise_std)
    return exposed_x

def interpret(rp, num_root, children, prob = 0.5):
    max_idx = np.argmax(rp[:num_root])
    max_val = rp[max_idx]
    last_max_idx = None
    while max_val > prob:
        last_max_idx = max_idx
        max_idx = children[max_idx][np.argmax(rp[children[max_idx]])]
        max_val = rp[max_idx]
        if len(children[max_idx]) == 0:
            last_max_idx = max_idx
            break
    return last_max_idx
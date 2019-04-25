import tensorflow as tf
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

def gen_ex(bs, p, c):
    '''
    Toy data generation function
    '''
    true_y = np.random.choice(c, size = bs)
    exposed_y = np.array([np.random.choice(p[i]) for i in true_y])
    true_x = np.array([np.sum([(k%5) * 5**-j for j,k in enumerate(p[i])]) for i in true_y])
    exposed_x = np.random.normal(true_x, .1, size = (16, bs)).T
    return exposed_x, exposed_y, true_x, true_y

def interpret(rp, num_root, children, prob = 0.5):
    max_idx = np.argmax(rp[:num_root])
    max_val = rp[max_idx]
    while max_val > prob:
        max_idx = np.argmax(rp[children[max_idx]])
        max_val = rp[max_idx]
    return max_idx

EPS = 1e-10

tree_dict = {'animal': {'cat': {'big-cat': {'lion': '', 'tiger': ''},
                                'small-cat': ''},
                        'dog': {'collie': '', 'dalmatian': '', 'terrier': ''},
                        'mouse': ''},
             'elements': {'acid': {'h2so4': '', 'hcl': ''},
                          'base': {'strong': {'koh': '', 'naoh': ''},
                                   'weak': {'ch3nh2': '', 'nh3': '', 'nh4oh': ''}}}}

class_list, n = get_dict_item(tree_dict)
num_root = len(n)
lenl = len(class_list)
n_flat = [num_root] + list(flatten(n))
subsoftmax_idx = np.cumsum([0] + n_flat, dtype = np.int32)

parents, children, childless = get_idxs(n_flat)
x_b, y_b, x_gt, y_gt = gen_ex(8, parents, childless)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape = [None, 16])
y = tf.placeholder(tf.int32, shape = [None])
y_onehot = tf.one_hot(y, lenl)

d1 = tf.layers.dense(x, 64, activation = tf.nn.relu)
d2 = tf.layers.dense(d1, 64, activation = tf.nn.relu)
d3 = tf.layers.dense(d2, 64, activation = tf.nn.relu)
d4 = tf.layers.dense(d3, 64, activation = tf.nn.relu)

logits = tf.layers.dense(x, lenl)

raw_probs = tf.concat([
        tf.nn.softmax(logits[:, subsoftmax_idx[i]: subsoftmax_idx[i + 1]]) \
        for i in range(len(n_flat))], 1)
probs = tf.concat([tf.reduce_prod(tf.gather(raw_probs, p, axis = 1),
                                  axis = 1,
                                  keepdims = True) for p in parents], 1)

loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.maximum(probs, EPS)) * y_onehot, 1))
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        x_b, y_b, x_gt, y_gt = gen_ex(8, parents, childless)
        _, l, rp = sess.run([train_op, loss, raw_probs], feed_dict = {x: x_b, y: y_b})
        print(l)
    for r, gt in zip(rp, y_gt):
        pred_class = interpret(r, num_root, children)
        print(class_list[pred_class], class_list[gt])
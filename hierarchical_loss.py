import tensorflow as tf
import numpy as np
from utils import get_dict_item, get_idxs, flatten, gen_ex, interpret

def wordtree_loss(logits, labels, word_tree, epsilon = 1e-5):
    '''
    Builds the wordtree style loss function as described in YOLO9000
    (https://arxiv.org/abs/1612.08242)

    Args:
        logits (tf.Tensor): Classification logits.
        labels (tf.Tensor): The one hot tensor of the ground truth labels.
        word_tree (dict): Dictionary of dictionaries showing the relationship between the classes.
        epsilon (float, optional): Epsilon term added to make the softmax cross entropy stable. Defaults to 1e-5.

    Returns:
        loss: Tensor of shape (batch_size, ), giving the loss for each example.
        raw_probs: The probability for each class (given its parents).
    '''
    _, n = get_dict_item(word_tree)
    n_flat = [len(n)] + list(flatten(n))
    parents, _, _ = get_idxs(n_flat)
    
    subsoftmax_idx = np.cumsum([0] + n_flat, dtype = np.int32)
    
    raw_probs = tf.concat([
        tf.nn.softmax(logits[:, subsoftmax_idx[i]: subsoftmax_idx[i + 1]]) \
        for i in range(len(n_flat))], 1)
    probs = tf.concat([tf.reduce_prod(tf.gather(raw_probs, p, axis = 1),
                                  axis = 1,
                                  keepdims = True) for p in parents], 1)
    
    loss = tf.reduce_sum(-tf.log(probs + epsilon) * labels, 1)
    return loss, raw_probs
    

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    word_tree = {'animal': {'cat': {'big-cat': {'lion': '', 'tiger': ''},
                                    'small-cat': ''},
                            'dog': {'collie': '', 'dalmatian': '', 'terrier': ''},
                            'mouse': ''},
                 'elements': {'acid': {'h2so4': '', 'hcl': ''},
                              'base': {'strong': {'koh': '', 'naoh': ''},
                                       'weak': {'ch3nh2': '', 'nh3': '', 'nh4oh': ''}}}}
    
    class_list, n = get_dict_item(word_tree)
    num_root = len(n)
    n_flat = [num_root] + list(flatten(n))
    n_classes = len(n_flat) - 1
    n_filters = 16
    
    parents, children, childless = get_idxs(n_flat)
    
    y_b = np.array(range(n_classes))
    num_ex = len(y_b)
    x_b = gen_ex(y_b, parents, children)
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape = [None, 16])
    y = tf.placeholder(tf.int32, shape = [None])
    y_onehot = tf.one_hot(y, n_classes)
    
    d1 = tf.layers.dense(x, n_filters, activation = tf.nn.relu)
    d2 = tf.layers.dense(d1, n_filters, activation = tf.nn.relu)
    
    logits = tf.layers.dense(d2, n_classes)
    
    _loss, raw_probs = wordtree_loss(logits = logits, labels = y_onehot, word_tree = word_tree)
    loss = tf.reduce_mean(_loss)

    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1000):
            idxs = np.random.randint(0, num_ex, 8)
            _, l, rp = sess.run([train_op, loss, raw_probs], feed_dict = {x: x_b[idxs], y: y_b[idxs]})
            if (step + 1)%100 == 0:
                print('Step {}: loss = {:.03e}'.format(step + 1, l))
    
        print('===TEST===')
        rp, = sess.run([raw_probs], feed_dict = {x: gen_ex(childless, parents, children, 0)})
        for r, gt in zip(rp, childless):
            pred_class = interpret(r, num_root, children)
            if not pred_class == gt:
                print('Truth: {}\t|\tPred: {}'.format(class_list[gt], class_list[pred_class]))
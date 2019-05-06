import numpy as np

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
"""Various utilities."""

import os
import csv

import torch
import random
import numpy as np

import socket
import datetime


#code by BDXC
def cmp_fast(grads, threshold):
    out = []
    sizes = []
    vecs = []
    shapes = []
    for t in grads:
        buf = t.flatten()
        sizes.append(len(buf))
        vecs.append(buf)
        shapes.append(t.shape)
        
    vec_g = torch.cat(vecs, 0)
    d = int(len(vec_g))
    # print("totle: "+ str(d))
    k = int(np.ceil(d*threshold))
    # print(k)

    indices = torch.abs(vec_g).topk(k)[1]
    out_g = torch.zeros_like(vec_g)
    out_g[indices] = vec_g[indices]
    out_g = torch.split(out_g, sizes, 0)
    for i in range(len(out_g)):
        buf=out_g[i].reshape(shapes[i])
        # for g in range(len(vec_g)):
        #     if abs(vec_g[g].data) <= threshold and vec_g[g].data != None:
        #         vec_g[g] = 0
        #         counter.update(1)
        #     else:
        #         counter.update(0)
        out.append(buf)
    return out

#code by BDXC
def get_indices(grads, threshold):
    out = []
    sizes = []
    vecs = []
    shapes = []
    for t in grads:
        buf = t.flatten()
        sizes.append(len(buf))
        vecs.append(buf)
        shapes.append(t.shape)
        
    vec_g = torch.cat(vecs, 0)
    d = int(len(vec_g))
    # print("totle: "+ str(d))
    k = int(np.ceil(d*threshold))
    # print(k)

    indices = torch.abs(vec_g).topk(k)[1]
    return indices

#code by BDXC
def cmp_rec(grads, indices):
    """
    accurate compress on the same vars as shared gradients
    """
    out = []
    sizes = []
    vecs = []
    shapes = []
    for t in grads:
        buf = t.flatten()
        sizes.append(len(buf))
        vecs.append(buf)
        shapes.append(t.shape)
        
    vec_g = torch.cat(vecs, 0)
    d = int(len(vec_g))
    # print("totle: "+ str(d))
    # print(k)

    ######compute similarity between two diffrent compress method#######
    # k = len(indices)/len(vec_g)
    # print (k)
    # tmp = torch.abs(vec_g).topk(len(indices))[1]
    # count = 0
    # for i in indices:
    #     if i in tmp:
    #         count+=1
    # print ('similarity:'+str(count/len(indices)))
    ###################################################################

    out_g = torch.zeros_like(vec_g)
    out_g[indices] = vec_g[indices]
    out_g = torch.split(out_g, sizes, 0)
    for i in range(len(out_g)):
        buf=out_g[i].reshape(shapes[i])
        # for g in range(len(vec_g)):
        #     if abs(vec_g[g].data) <= threshold and vec_g[g].data != None:
        #         vec_g[g] = 0
        #         counter.update(1)
        #     else:
        #         counter.update(0)
        out.append(buf)
    return out

def system_startup(args=None, defs=None):
    """Print useful system information."""
    # Choose GPU device and print status information:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return setup

def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')
        print(f'Would save these keys: {fieldnames}.')

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

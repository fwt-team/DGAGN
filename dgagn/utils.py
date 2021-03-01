# encoding: utf-8
try:
    import torch
    import argparse
    import numpy as np
    import torch.nn as nn
    import torch_geometric.nn as gnn

    from scipy.optimize import linear_sum_assignment as linear_assignment

    from dgagn.config import DEVICE
except ImportError as e:
    print(e)
    raise ImportError


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_weights(net):

    for m in net.modules():
        if isinstance(m, gnn.GCNConv):
            m.weight.data.normal_(0, 0.02)
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, gnn.GCNConv):
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight)
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def cluster_acc(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total*1.0/Y_pred.size, w


def besseli(nu, x):

    frac = x / nu
    square = 1 + frac**2
    root = torch.sqrt(square)
    eta = root + torch.log(frac) - torch.log(1 + root)
    approx = -torch.log(torch.sqrt(torch.tensor(2 * np.pi * nu, dtype=torch.float).to(DEVICE))) + nu * eta - 0.25*torch.log(square)

    return torch.exp(approx)


def d_besseli(nu, kk):

    try:
        bes = besseli(nu + 1, kk) / (besseli(nu, kk) + 1e-10)
        assert (min(torch.isfinite(bes)))
    except:
        bes = torch.sqrt(1 + (nu**2) / (kk**2))

    return bes


def exchange(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]

    return total * 1.0 / Y_pred.size, ind


def find_repeat(x, y):

    return np.array([i for i in x if i in y])

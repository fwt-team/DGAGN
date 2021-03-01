# encoding: utf-8
try:
    import os.path as osp
    import numpy as np
    import torch_geometric.transforms as T

    from torch_geometric.datasets import Planetoid
except ImportError as e:
    print(e)
    raise ImportError


dataset_list = ['Cora', 'Citeseer', 'Pubmed']


def get_dif_ran(length: int, test_len: int) -> list:

    ran_index = []
    while True:
        if len(ran_index) == length - test_len:
            break

        r = np.random.randint(0, length)
        if r not in ran_index:
            ran_index.append(r)

    return ran_index


def get_data(name='Cora', split=False, rate=0):

    dataset = name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset.get(0)
    data.train_mask[:] = False
    data.test_mask[:] = True
    if split:
        leng = data.x.size(0)
        test_len = int(leng * rate)
        ran_index = get_dif_ran(leng, test_len)

        data.train_mask[ran_index] = True
        data.test_mask[ran_index] = False
        return data

    return data.x, data.y, data.edge_index

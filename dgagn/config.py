# encoding: utf-8
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(DEVICE))
# Local directory of CypherCat API
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(ROOT_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# difference datasets config
# pretrain_batch_size, train_batch_size, latent_dim, hidden_channels, output_channels, all_data_size, pre_epoch, pre_lr, train_lr,
# lam_kl, lam_rec, lam_rec1, lam_adv, lowest_kappa, n_cluster
DATA_PARAMS = {
    'Cora': (
        2708, 2708, 5, 100, 1433, 2708, 500, 1e-2, 2e-2, 1, 3, 15, 7, 90, 7,
    ),
}

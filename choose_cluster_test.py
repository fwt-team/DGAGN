# -*- coding: utf-8 -*-
try:
    import os
    import argparse
    import torch

    from vmfmix.vmf import VMFMixture
    from dgagn.datasets import dataset_list, get_data
    from dgagn.config import RUNS_DIR, DEVICE, DATA_PARAMS
    from dgagn.model import Encoder
    from dgagn.utils import cluster_acc
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="DGAGN", help="Name of training run")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='Cora', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="1")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, 'VMFMM_V{}'.format(args.version_name))
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    data_params = DATA_PARAMS[dataset_name]
    pretrain_batch_size, train_batch_size, latent_dim, hidden_channels, output_channels, data_size, \
    pre_epoch, pre_lr, train_lr, lam_kl, lam_rec, lam_rec1, lam_adv, lowest_kappa, n_cluster = data_params
    print('params: {}'.format(data_params))

    # net
    encoder = Encoder(input_channels=output_channels, output_channels=latent_dim, hidden_channels=hidden_channels, r=lowest_kappa)

    # set device: cuda or cpu
    encoder.to(DEVICE)

    x, y, edge_index = get_data(args.dataset_name)
    x, y, edge_index = x.to(DEVICE), y.to(DEVICE), edge_index.to(DEVICE)

    # =============================================================== #
    # ==========================pretraining========================== #
    # =============================================================== #
    pre_train_path = os.path.join(models_dir, 'pre_train')
    print("load pretrain model...")
    encoder.load_state_dict(torch.load(os.path.join(pre_train_path, "enc.pkl"), map_location=DEVICE))

    print("choose the best number of cluster...")
    for i in range(1, 11):

        _vmfmm = VMFMixture(n_cluster=i, max_iter=100)
        Z = []
        Y = []
        with torch.no_grad():
            _, z, _ = encoder(x, edge_index)
            Z.append(z)
            Y.append(y)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().cpu().numpy()

        _vmfmm.fit(Z)
        pre = _vmfmm.predict(Z)
        print('Current number of cluster={}, Acc={:.4f}%'.format(i, cluster_acc(pre, Y)[0] * 100))


if __name__ == '__main__':
    main()

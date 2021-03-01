# -*- coding: utf-8 -*-
try:
    import os
    import argparse
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score

    from train_mm import main as MM
    from train_classifier import Classifier
    from dgagn.datasets import get_data, dataset_list
    from dgagn.config import DATA_PARAMS, RUNS_DIR, DEVICE, DATASETS_DIR
    from dgagn.utils import exchange, find_repeat

except Exception as e:
    print(e)
    raise Exception


def main():

    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="DGAGN", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='Cora', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="1")
    parser.add_argument("-p", "--pretrain", dest="pretrain", default=1)
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, 'VMFMM_V{}'.format(args.version_name))
    data_dir = os.path.join(DATASETS_DIR, dataset_name)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    data_params = DATA_PARAMS[dataset_name]
    data = get_data(dataset_name, split=True, rate=0.99)
    x, y, edge_index, train_mask, test_mask = data.x, data.y, data.edge_index, data.train_mask, data.test_mask
    x, y, edge_index = x.to(DEVICE), y.to(DEVICE), edge_index.to(DEVICE)

    encoder, _, vmfmm, _ = MM(x, y, edge_index, data_params, run_dir, n_epochs=300)
    classifier = Classifier(x, y, edge_index, train_mask, test_mask, data_params, run_dir)

    train_len = train_mask[train_mask == True].size(0)

    same_index = None
    exchanged_y = torch.clone(y)
    for i in range(35):
        cluster_logit, cluster_pred = vmfmm.predict_logit(encoder(x, edge_index)[0])
        acc, ind = exchange(y.data.cpu().numpy(), cluster_pred)
        pred_c, logit = classifier.train(exchanged_y, same_index)
        acc_c = accuracy_score(y.data.cpu().numpy()[test_mask], pred_c[test_mask])

        same_index = []
        for j, k in zip(ind[0], ind[1]):
            t1 = np.where(cluster_pred[test_mask] == k)[0]
            t2 = np.where(logit[test_mask][pred_c[test_mask] == j] > 0.999999)[0]
            # if len(t1) > 0:
            #     ran_index = np.random.randint(0, len(t1), size=100)
            #     t1 = t1[ran_index]
            if len(t2) > 0:
                ran_index = np.random.randint(0, len(t2), size=100)
                t2 = t2[ran_index]
            repeat = find_repeat(t2, t1)

            same_index.append(repeat)

        same_index = np.unique(np.concatenate(same_index).astype(np.int))

        if len(same_index) > 0:
            same_index = same_index + train_len
            exchanged_y[same_index] = torch.tensor(pred_c[same_index], device=DEVICE)

            print('clsuter_acc: {}, acc: {}'.format(acc, acc_c))


if __name__ == '__main__':

    main()

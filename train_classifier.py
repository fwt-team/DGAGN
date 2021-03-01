# -*- coding: utf-8 -*-
try:
    import torch
    import os
    import torch.nn as nn

    from itertools import chain
    from torch.optim import Adam
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    from dgagn.model import Encoder
    from dgagn.config import DEVICE, RUNS_DIR

except Exception as e:
    print(e)
    raise Exception


class Classifier:

    def __init__(self, x, y, edge_index, train_mask, test_mask, data_params, run_dir):

        self.x, self.y, self.edge_index, self.train_mask, self.test_mask, self.data_params, self.run_dir \
            = x, y, edge_index, train_mask, test_mask, data_params, run_dir

        self.models_dir = os.path.join(run_dir, 'models')
        self.encoder, self.xe_loss, self.optim = None, None, None
        self.init_encoder()

    def init_encoder(self):

        pretrain_batch_size, train_batch_size, latent_dim, hidden_channels, output_channels, data_size, \
        pre_epoch, pre_lr, train_lr, lam_kl, lam_rec, lam_rec1, lam_adv, lowest_kappa, n_cluster = self.data_params
        encoder = Encoder(input_channels=output_channels, output_channels=latent_dim,
                          hidden_channels=hidden_channels,
                          r=lowest_kappa)
        encoder.to(DEVICE)
        encoder.load_state_dict(torch.load(os.path.join(self.models_dir, 'enc.pkl'), map_location=DEVICE))
        encoder.init_classifier(n_cluster)
        self.encoder = encoder

        self.xe_loss = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
        self.optim = Adam(chain(
            encoder.cla.parameters(),
            encoder.conv1.parameters(),
        ), lr=1e-3, betas=(0.3, 0.5))

    def train(self, u_y, same_index=None, epochs=500):

        if same_index is not None:
            self.train_mask[same_index] = True
            self.init_encoder()

            print('other: {}'.format(len(self.train_mask[self.train_mask == True])))
        encoder = self.encoder
        optim = self.optim
        train_mask = self.train_mask
        test_mask = self.test_mask
        x = self.x
        y = self.y
        edge_index = self.edge_index
        train_len = train_mask[train_mask == True].size(0)
        best_score = 0
        best_model = None
        for epoch in range(epochs):

            encoder.train()
            optim.zero_grad()

            logit, pred = encoder.classifier(x, edge_index)

            loss = self.xe_loss(logit[train_mask], u_y[train_mask]) / train_len
            loss.backward()
            optim.step()

            # valid
            encoder.eval()
            with torch.no_grad():
                _, pred = encoder.classifier(x, edge_index)
                target, pred = y[test_mask].data.cpu().numpy(), pred[test_mask].data.cpu().numpy()
                acc = accuracy_score(target, pred)
                p = precision_score(target, pred, average='macro')
                r = recall_score(target, pred, average='macro')
                f1 = f1_score(target, pred, average='macro')

                if acc > best_score:
                    best_score = acc
                    best_model = encoder.state_dict()
                # print('iter: {}, loss: {:.4f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}'.
                #       format(epoch, loss, acc, p, r, f1))

        torch.save(best_model, os.path.join(self.models_dir, 'classifier.pkl'))
        print('best_acc is: {:.4f}'.format(best_score))
        encoder.load_state_dict(best_model)
        logit, pred = encoder.classifier(x, edge_index)
        return pred.data.cpu().numpy(), logit.data.cpu().numpy()


if __name__ == '__main__':

    from dgagn.datasets import get_data
    from dgagn.config import DATA_PARAMS

    run_dir = os.path.join(RUNS_DIR, 'Cora', 'DGAGN', 'VMFMM_V1')
    data_params = DATA_PARAMS['Cora']
    data = get_data('Cora', split=True, rate=0.99)
    x, y, edge_index, train_mask, test_mask = data.x, data.y, data.edge_index, data.train_mask, data.test_mask
    x, y, edge_index = x.to(DEVICE), y.to(DEVICE), edge_index.to(DEVICE)
    classifier = Classifier(x, y, edge_index, train_mask, test_mask, data_params, run_dir)
    classifier.train(y)

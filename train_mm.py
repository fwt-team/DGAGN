# -*- coding: utf-8 -*-
try:
    import os
    import time
    from itertools import chain
    import torch
    import torch.nn as nn

    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

    from vmfmix.vmf import VMFMixture
    from dgagn.config import RUNS_DIR, DEVICE
    from dgagn.model import Generator, VMFMM, Encoder, Discriminator
    from dgagn.utils import cluster_acc

except ImportError as e:
    print(e)
    raise ImportError


def main(x, y, edge_index, data_params, run_dir, n_epochs=500, load=True):

    # make directory
    models_dir = os.path.join(run_dir, 'models')
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # -----hyper-parameters-----
    b1 = 0.3
    b2 = 0.5
    decay = 2.5 * 1e-5
    print('b1: {}, b2: {}, decay: {}'.format(b1, b2, decay))

    pretrain_batch_size, train_batch_size, latent_dim, hidden_channels, output_channels, data_size, \
    pre_epoch, pre_lr, train_lr, lam_kl, lam_rec, lam_rec1, lam_adv, lowest_kappa, n_cluster = data_params
    print('params: {}'.format(data_params))

    # net
    gen = Generator(latent_dim=latent_dim, hidden_channels=hidden_channels, output_channels=output_channels)
    vmfmm = VMFMM(n_cluster=n_cluster, n_features=latent_dim)
    encoder = Encoder(input_channels=output_channels, output_channels=latent_dim, hidden_channels=hidden_channels, r=lowest_kappa)
    dis = Discriminator(input_channels=output_channels, output_channels=latent_dim, hidden_channels=hidden_channels)
    gen.to(DEVICE)
    encoder.to(DEVICE)
    vmfmm.to(DEVICE)
    dis.to(DEVICE)

    if os.path.exists(os.path.join(models_dir, 'enc.pkl')) and load:

        gen.load_state_dict(torch.load(os.path.join(models_dir, 'gen.pkl'), map_location=DEVICE))
        vmfmm.load_state_dict(torch.load(os.path.join(models_dir, 'vmfmm.pkl'), map_location=DEVICE))
        encoder.load_state_dict(torch.load(os.path.join(models_dir, 'enc.pkl'), map_location=DEVICE))
        dis.load_state_dict(torch.load(os.path.join(models_dir, 'dis.pkl'), map_location=DEVICE))

        return encoder, gen, vmfmm, dis

    xe_loss = nn.BCELoss(reduction="sum").to(DEVICE)
    xe_loss_logit = nn.BCEWithLogitsLoss(reduction="sum").to(DEVICE)

    # optimization
    gen_enc_ops = torch.optim.Adam(chain(
        gen.parameters(),
        encoder.parameters(),
    ), lr=pre_lr, betas=(0.5, 0.99), weight_decay=decay)
    gen_enc_vmfmm_ops = torch.optim.Adam(chain(
        gen.parameters(),
        encoder.parameters(),
    ), lr=train_lr, betas=(b1, b2))
    dis_ops = torch.optim.Adam(dis.parameters(), lr=train_lr * 0.2, betas=(b1, b2))
    vmf_ops = torch.optim.Adam(chain(
        vmfmm.parameters()
    ), lr=train_lr * 0.2, betas=(b1, b2))

    lr_s = StepLR(gen_enc_vmfmm_ops, step_size=20, gamma=1)

    x, y, edge_index = x.to(DEVICE), y.to(DEVICE), edge_index.to(DEVICE)

    # =============================================================== #
    # ==========================pretraining========================== #
    # =============================================================== #
    pre_train_path = os.path.join(models_dir, 'pre_train')
    if not os.path.exists(pre_train_path):

        print('Pretraining......')
        epoch_bar = tqdm(range(pre_epoch))
        for _ in epoch_bar:

            _, z, _ = encoder(x, edge_index, False)
            x_, _ = gen(z, edge_index, False)
            loss = xe_loss(x_, x) / pretrain_batch_size + gen.rec_loss(z, edge_index)

            L = loss.detach().cpu().numpy()

            gen_enc_ops.zero_grad()
            loss.backward()
            gen_enc_ops.step()

            # epoch_bar.write('Loss={:.4f}'.format(L))

        print("choose the best number of cluster...")
        best_score = 0
        best_model = None
        for i in range(1, 11):

            best_vmfmm = None
            best_pacc = 0
            for j in range(20):
                Z = []
                Y = []
                with torch.no_grad():
                    _, z, _ = encoder(x, edge_index)
                    Z.append(z)
                    Y.append(y)

                Z = torch.cat(Z, 0).detach().cpu().numpy()
                Y = torch.cat(Y, 0).detach().cpu().numpy()
                _vmfmm = VMFMixture(n_cluster=i, max_iter=100)
                _vmfmm.fit(Z)
                pre = _vmfmm.predict(Z)
                acc = cluster_acc(pre, Y)[0] * 100
                if best_pacc < acc:
                    best_pacc = acc
                    best_vmfmm = _vmfmm

            print('Current number of cluster={}, Acc={:.4f}%'.format(i, best_pacc))
            if best_score < best_pacc:
                best_score = best_pacc
                n_cluster = i
                best_model = best_vmfmm

        print('best number of cluster is: {}'.format(n_cluster))
        vmfmm.reset_n_cluster(n_cluster)
        vmfmm.to(DEVICE)

        vmfmm.pi_.data = torch.from_numpy(best_model.pi).to(DEVICE).float()
        vmfmm.mu_c.data = torch.from_numpy(best_model.xi).to(DEVICE).float()
        vmfmm.k_c.data = torch.from_numpy(best_model.k).to(DEVICE).float()

        os.makedirs(pre_train_path, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(pre_train_path, 'enc.pkl'))
        torch.save(gen.state_dict(), os.path.join(pre_train_path, 'gen.pkl'))
        torch.save(vmfmm.state_dict(), os.path.join(pre_train_path, 'vmfmm.pkl'))

    else:
        print("load pretrain model...")
        gen.load_state_dict(torch.load(os.path.join(pre_train_path, "gen.pkl"), map_location=DEVICE))
        vmfmm.load_state_dict(torch.load(os.path.join(pre_train_path, "vmfmm.pkl"), map_location=DEVICE))
        encoder.load_state_dict(torch.load(os.path.join(pre_train_path, "enc.pkl"), map_location=DEVICE))

    # =============================================================== #
    # ============================training=========================== #
    # =============================================================== #
    epoch_bar = tqdm(range(0, n_epochs))
    best_score, best_nmi, best_ari = 0, 0, 0
    best_epoch = 0
    best_gen, best_encoder, best_vmf, best_dis = None, None, None, None

    logger = open(os.path.join(log_path, "log.txt"), 'a')
    logger.write(
        "===============================================================\n"
        "============================Beginning==========================\n"
        "===============================================================\n"
    )
    logger.close()
    begin = time.time()
    for epoch in epoch_bar:
        g_t_loss = 0

        gen.train()
        vmfmm.train()
        encoder.train()
        gen_enc_vmfmm_ops.zero_grad()
        vmf_ops.zero_grad()
        dis_ops.zero_grad()

        z, mu, k = encoder(x, edge_index)

        fake_x, _ = gen(z, edge_index)
        fake_x_n, _ = gen(mu, edge_index)

        D_real = dis(x, edge_index)
        D_fake = dis(fake_x, edge_index)
        D_fake_n = dis(fake_x_n, edge_index)

        d_loss = 2 * torch.mean(D_real) - torch.mean(D_fake) - torch.mean(D_fake_n)
        d_loss.backward(retain_graph=True)
        dis_ops.step()

        rec_loss = xe_loss(fake_x, x) / train_batch_size + gen.rec_loss(z, edge_index)
        rec_loss1 = xe_loss_logit(fake_x, fake_x_n) / train_batch_size
        # train generator, encoder and vmfmm
        g_loss = lam_kl * vmfmm.vmfmm_Loss(z, mu, k) + lam_rec * rec_loss + lam_rec1 * rec_loss1 + \
                 lam_adv * (torch.mean(D_fake) + torch.mean(D_fake_n))

        g_loss.backward()

        nn.utils.clip_grad_norm_(chain(
            vmfmm.parameters(),
            encoder.parameters(),
            gen.parameters(),
            dis.parameters(),
        ), 8)

        gen_enc_vmfmm_ops.step()
        vmf_ops.step()

        g_t_loss += g_loss
        vmfmm.mu_c.data = vmfmm.mu_c.data / torch.norm(vmfmm.mu_c.data, dim=1, keepdim=True)
        lr_s.step()
        print(vmfmm.vmfmm_Loss(z, mu, k))

        # =============================================================== #
        # ==============================test============================= #
        # =============================================================== #
        encoder.eval()
        vmfmm.eval()
        with torch.no_grad():
            _data, _target = x, y
            _target = _target.data.cpu().numpy()
            _z, _, _ = encoder(_data, edge_index)
            _pred = vmfmm.predict(_z)
            _acc = cluster_acc(_pred, _target)[0] * 100
            _nmi = NMI(_pred, _target)
            _ari = ARI(_target, _pred)

            if best_score < _acc:
                best_score, best_nmi, best_ari = _acc, _nmi, _ari
                best_epoch = epoch
                best_gen = gen.state_dict()
                best_encoder = encoder.state_dict()
                best_vmf = vmfmm.state_dict()
                best_dis = dis.state_dict()

            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "[DGAGN]: epoch: {}, g_loss: {:.3f}, acc: {:.4f}%, nmi: {:.4f}, ari: {:.4f}\n".format(epoch,
                                                                       g_t_loss, _acc, _nmi, _ari)
            )
            logger.close()
            print("[DGAGN]: epoch: {}, g_loss: {}, acc: {:.4f}%, nmi: {:.4f}, ari: {:.4f}".format(epoch,
                                                                       g_t_loss, _acc, _nmi, _ari))

    end = time.time()
    logger = open(os.path.join(log_path, "log.txt"), 'a')
    logger.write(
        "best acc is: {:.4f}, nmi is: {:.4f}, ari is: {:.4f}, iteration is: {}, runing times is: {:.4f}\n".format(
            best_score, best_nmi, best_ari, best_epoch, end - begin)
    )
    logger.write(
        "===============================================================\n"
        "=============================Ending============================\n"
        "===============================================================\n\n\n"
    )
    logger.close()

    print('best acc is: {:.4f}, nmi is: {:.4f}, ari is: {:.4f}, iteration is: {}'.format(best_score, best_nmi, best_ari, best_epoch))
    print('save model......')
    torch.save(best_gen, os.path.join(models_dir, 'gen.pkl'))
    torch.save(best_encoder, os.path.join(models_dir, 'enc.pkl'))
    torch.save(best_vmf, os.path.join(models_dir, 'vmfmm.pkl'))
    torch.save(best_dis, os.path.join(models_dir, 'dis.pkl'))

    return encoder, gen, vmfmm, dis


if __name__ == '__main__':

    from dgagn.datasets import get_data
    from dgagn.config import DATA_PARAMS

    data_name = 'Cora'
    run_dir = os.path.join(RUNS_DIR, data_name, 'DGAGN', 'VMFMM_V1')
    data_params = DATA_PARAMS[data_name]
    data = get_data(data_name)
    x, y, edge_index = data
    x, y, edge_index = x.to(DEVICE), y.to(DEVICE), edge_index.to(DEVICE)
    main(x, y, edge_index, data_params, run_dir, n_epochs=300, load=False)

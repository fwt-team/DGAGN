# -*- coding: utf-8 -*-
try:
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                       add_self_loops)

    from dgagn.utils import init_weights, d_besseli, besseli
    from dgagn.config import DEVICE
    from vmfmix.von_mises_fisher import VonMisesFisher

except ImportError as e:
    print(e)
    raise ImportError


class Generator(nn.Module):

    def __init__(self, latent_dim=50, hidden_channels=100, output_channels=1200, verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.verbose = verbose

        self.conv1 = GCNConv(latent_dim, hidden_channels, cached=True)
        self.conv_X = GCNConv(hidden_channels, output_channels, cached=True)
        self.conv_A = GCNConv(hidden_channels, latent_dim, cached=True)
        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, z, edge_index, dropout=True):

        if dropout:
            z = torch.dropout(z, p=0.5, train=True)
        x = F.relu(self.conv1(z, edge_index))
        A = self.conv_A(x, edge_index)
        A = torch.sigmoid((A[edge_index[0]] * A[edge_index[1]]).sum(dim=1))
        x = torch.sigmoid(self.conv_X(x, edge_index))
        return x, A

    def rec_loss(self, z, pos_edge_index):
        EPS = 1e-15
        pos_loss = -torch.log(
            self.forward(z, pos_edge_index)[1] + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.forward(z, neg_edge_index)[1] +
                              EPS).mean()

        return pos_loss + neg_loss


class Encoder(nn.Module):

    def __init__(self, input_channels=1, hidden_channels=100, output_channels=64, r=10, verbose=False):
        super(Encoder, self).__init__()

        self.cla = None
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.r = r
        self.verbose = verbose

        self.conv1 = GCNConv(input_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, output_channels, cached=True)
        self.conv_k = GCNConv(hidden_channels, 1, cached=True)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def init_classifier(self, n_cluster):

        self.cla = GCNConv(self.hidden_channels, n_cluster, cached=True).to(DEVICE)
        init_weights(self.cla)

    def classifier(self, x, edge_index, dropout=True):

        if dropout:
            x = torch.dropout(x, p=0.1, train=True)
        x = F.relu(self.conv1(x, edge_index))
        logit = torch.softmax(self.cla(x, edge_index), dim=1)
        pred = torch.argmax(logit, dim=1)

        return logit, pred

    def forward(self, x, edge_index, dropout=True):

        if dropout:
            x = torch.dropout(x, p=0.1, train=True)
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        k = F.softplus(self.conv_k(x, edge_index)) + self.r

        mu = mu / mu.norm(dim=1, keepdim=True)
        z = VonMisesFisher(mu, k).rsample()

        return z, mu, k


class Discriminator(nn.Module):

    def __init__(self, input_channels=1, hidden_channels=100, output_channels=64, verbose=False):
        super(Discriminator, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.verbose = verbose

        self.conv1 = GCNConv(input_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, output_channels, cached=True)
        self.conv3 = GCNConv(output_channels, 1, cached=True)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x, edge_index):

        x = torch.dropout(x, p=0.5, train=True)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        return x


class VMFMM(nn.Module):

    def __init__(self, n_cluster=10, n_features=10):
        super(VMFMM, self).__init__()

        self.n_cluster = n_cluster
        self.n_features = n_features
        self.pi_ = None
        self.mu_c = None
        self.k_c = None
        self.init_params()

    def init_params(self):

        mu = torch.FloatTensor(self.n_cluster, self.n_features).normal_(0, 0.02)
        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(mu / mu.norm(dim=-1, keepdim=True), requires_grad=True)
        self.k_c = nn.Parameter(torch.FloatTensor(self.n_cluster, ).uniform_(1, 5), requires_grad=True)

    def reset_n_cluster(self, n_cluster):

        self.n_cluster = n_cluster
        self.init_params()

    def predict_logit(self, z):

        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c))

        yita = torch.softmax(yita_c, dim=1).data.cpu().numpy()
        pred = np.argmax(yita, axis=1)

        return yita, pred

    def predict(self, z):

        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def sample_by_k(self, k, num=10):

        mu = self.mu_c[k:k+1]
        k = self.k_c[k].view((1, 1))
        z = None
        for i in range(num):
            _z = VonMisesFisher(mu, k).rsample()
            if z is None:
                z = _z
            else:
                z = torch.cat((z, _z))
        return z

    def vmfmm_pdfs_log(self, x, mu_c, k_c):

        VMF = []
        for c in range(self.n_cluster):
            VMF.append(self.vmfmm_pdf_log(x, mu_c[c:c + 1, :], k_c[c]).view(-1, 1))
        return torch.cat(VMF, 1)

    @staticmethod
    def vmfmm_pdf_log(x, mu, k):

        D = x.size(1)
        log_pdf = (D / 2 - 1) * torch.log(k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k)) \
                  + x.mm(torch.transpose(mu, 1, 0) * k)
        return log_pdf

    def vmfmm_Loss(self, z, z_mu, z_k):

        det = 1e-10
        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c

        D = self.n_features
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

        # batch * n_cluster
        e_k_mu_z = (d_besseli(D / 2 - 1, z_k) * z_mu).mm((k_c.unsqueeze(1) * mu_c).transpose(1, 0))

        # batch * 1
        e_k_mu_z_new = torch.sum((d_besseli(D / 2 - 1, z_k) * z_mu) * (z_k * z_mu), 1, keepdim=True)

        # e_log_z_x
        Loss = torch.mean((D * ((D / 2 - 1) * torch.log(z_k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, z_k)) + e_k_mu_z_new)))

        # e_log_z_c
        Loss -= torch.mean(torch.sum(yita_c * (
                D * ((D / 2 - 1) * torch.log(k_c) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k_c)) + e_k_mu_z)), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1))
        return Loss

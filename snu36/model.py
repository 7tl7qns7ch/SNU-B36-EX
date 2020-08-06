import torch.nn as nn
import torch.nn.functional as F
import math
import torch

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np

from time import time
from math import sqrt


##### Convolutional neural network #####

class ConvNet(nn.Module):
    def __init__(self, num_classes=135, feature_size=128, is_urban=False):
        super(ConvNet, self).__init__()
        self.is_urban = is_urban
        self.layer1 = nn.Sequential(
            nn.Conv1d(120, feature_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Conv1d(feature_size, feature_size, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Conv1d(feature_size, feature_size, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0))
        self.fc1 = nn.Sequential(
            nn.Linear(feature_size * 27, feature_size),
            nn.ELU())
        self.fc2 = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ELU())
        self.fc3 = nn.Linear(feature_size, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        prelogits = self.fc2(out)
        logits = self.fc3(out)
        if self.is_urban:
            logits = self.sigmoid(logits)

        return prelogits, logits


class sparse_NMF(nn.Module):
    def __init__(self, Xshape, n_components=None):
        super().__init__()
        if not n_components:
            self.n_components = self.K
        else:
            self.n_components = n_components
        self.K, self.M = Xshape
        self.W = torch.rand(self.K, self.n_components, dtype=torch.float64).cuda()
        self.H = torch.rand(self.n_components, self.M, dtype=torch.float64).cuda()

    def fit(self, X, W=None, update_W=True, lamb=0.01, max_iter=300, myeps=10E-100):
        if update_W == True:
            if W is not None:
                self.W.data.copy_(W)
            else:
                pass
            self.W = self.W / torch.sum(self.W ** 2, dim=0) ** (.5)

        else:
            if W is not None:
                self.W.data.copy_(W)
            else:
                pass

        self.re_error = []
        self.sp_error = []
        self.error = []

        start_time = time()
        for n_iter in range(1, max_iter + 1):
            self.H = self.H * ((self.W.t() @ X) / (((self.W.t() @ self.W) @ self.H) + lamb + myeps))

            if update_W == True:
                self.W = self.W * ((X @ self.H.t() + self.W * (
                    (torch.ones((X.shape[0], X.shape[0]), dtype=torch.float64).cuda() @ (
                                self.W @ (self.H @ self.H.t())) * self.W))) / (
                                           (self.W @ (self.H @ self.H.t())) + self.W * (
                                       (torch.ones((X.shape[0], X.shape[0]), dtype=torch.float64).cuda() @ (
                                                   X @ self.H.t()) * self.W)) + myeps))

                self.W = self.W / torch.sum(self.W ** 2, dim=0) ** (.5)

            else:
                pass

            re_error = 0.5 * (torch.norm(X - (self.W @ self.H))) ** 2
            re_error = re_error.detach().cpu()
            self.re_error.append(re_error)

            sp_error = lamb * torch.sum(torch.sum(self.H))
            sp_error = sp_error.detach().cpu()
            self.sp_error.append(sp_error)

            error = re_error + sp_error
            self.error.append(error)

            if n_iter % 10 == 0:
                iter_time = time()
                print("Epoch %02d reached after %.3f seconds, error: %f, re_error: %f, sp_error: %f" % (n_iter, iter_time - start_time, error, re_error, sp_error))


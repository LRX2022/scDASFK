from __future__ import print_function, division
from  utils import distance
import torch
import torch.nn as nn



class CN(nn.Module):
    def __init__(self,device="cuda",dropouts=0.1,input =512):
        super(CN, self).__init__()
        self.sigma = 0.01
        self.device =device
        self.ru = nn.LeakyReLU()  #
        self.drop = nn.Dropout(p=dropouts)
        self.enc1 = nn.Linear(input, input)
        self.dec1 = nn.Linear(input, input)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)
                m.bias.data.zero_()
    def setC(self,c):
        self.centroids=c

    def netnoise(self, x):
        x = x + (torch.rand(x.shape)*1).to(device=self.device)
        enc1 = self.enc1(x)
        z = self.ru(enc1)
        dec1 = self.dec1(z)
        dec1 = self.drop(dec1)
        dec1 = self.ru(dec1)
        return z, dec1

    def netclean(self, x):
        enc1 = self.enc1(x)
        z = self.ru(enc1)
        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        return z, dec1

    def _build_lossCN(self, x, zclean, d, u):
        size = x.shape[0]
        t = d * u
        distances = distance(zclean.t(), self.centroids)
        loss1 =  torch.trace(distances.t().matmul(t)) / (size)  #.matmul(t)乘法，torch.trace求对角线之和
        return loss1

    def _update_D(self, Z):
        if self.sigma is None:
            return torch.ones([Z.shape[1], self.centroids.shape[1]]).to(self.device)
        else:
            distances = distance(Z, self.centroids, False)
            return (1 + self.sigma) * (distances + 2 * self.sigma) / (2 * (distances + self.sigma))

    def clustering(self, Z):
        D = self._update_D(Z)
        T = D *self.U
        self.centroids = Z.matmul(T) / T.sum(dim=0).reshape([1, -1])
        self._update_U(Z)
        _, y_pred = self.U.max(dim=1)
        y_pred = y_pred.detach().cpu() + 1
        y_pred = y_pred.numpy()
        return y_pred

    def _update_U(self, Z):
        if self.sigma is None:
            distances = distance(Z, self.centroids, False)
        else:
            distances = self.adaptive_loss(distance(Z, self.centroids, False), self.sigma)
        U = torch.exp(-distances)
        self.U = U / U.sum(dim=1).reshape([-1, 1])

    def adaptive_loss(self,D, sigma):
        return (1 + sigma) * D * D / (D + sigma)
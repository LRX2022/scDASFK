from __future__ import print_function, division
import math
from preprocessing import computeCentroids
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from SelfAttention import SelfAttention
from DN import DN
from CN import CN
from sklearn.cluster import AgglomerativeClustering


class scDASFK(nn.Module):
    def __init__(self,
                 dropouts=0.1,
                 input =3000,
                 output=1000,
                 labels =None,
                 X=None,
                 lr =0.0001,
                 device =None,
                 num_attention_heads=2):
        super(scDASFK, self).__init__()
        self.labels = labels
        self.X = X
        self.n_clusters =len(np.unique(self.labels))
        self.lr = lr
        self.drop =dropouts
        self.device = device
        if (self.X.shape[0] < 500):
            self.batch_size = 32
        elif (self.X.shape[0] < 700):
            self.batch_size = 100
        else:
            self.batch_size = 128
        self.SelfAttention = SelfAttention(num_attention_heads=num_attention_heads, input_size=output, hidden_size=output)
        self.DN = DN(device=self.device,dropouts=self.drop,input =input,output =output)
        self.CN =CN(device=self.device,dropouts=self.drop,input =output)

    def runDN(self,eD,train_loaderDN,optimizerDN):
        for epoch1 in range(eD):
            loss = 0
            c = 0

            for i, batch in enumerate(train_loaderDN):
                x = batch[0][0]
                idx = batch[1]
                optimizerDN.zero_grad()  # 梯度置零
                znoise, xdec1noise = self.DN.netnoise(x)
                zclean, xdec1clean = self.DN.netclean(x)
                u = self.CN.U[idx, :]
                loss0 = self.DN._build_lossDN(x, zclean, znoise, xdec1noise, u)
                if (math.isnan(loss0.item())):
                    break
                c = c + 1
                loss = loss + loss0.item()
                loss0.backward()
                optimizerDN.step()
            # loss11 = np.append(loss11, loss)
            # if (len(loss11) % 50 == 0):
            #     #print(np.std(loss11))
            #     if (np.std(loss11) < 0.03): break
            #     loss11 = np.array([])

    def runCN(self,eF,train_loaderCN,optimizerCN):
        y_pred=None
        for epoch0 in range(eF):
            loss = 0
            c = 0
            D = self.CN._update_D(self.Z)

            for i, batch in enumerate(train_loaderCN):
                x = batch[0][0]
                idx = batch[1]
                optimizerCN.zero_grad()  # 梯度置零
                znoise, xdec1noise = self.CN.netnoise(x)
                pin = self.SelfAttention(torch.stack([x, znoise, xdec1noise]))
                #pin = znoise + xdec1noise + x
                d = D[idx, :]
                u = self.CN.U[idx, :]
                loss1 = self.CN._build_lossCN(x, pin, d, u)
                c = c + 1
                loss = loss + loss1.item()
                loss1.backward()
                optimizerCN.step()

            z, xdec1 = self.CN.netclean(self.zz)
            pinjie = self.SelfAttention(torch.stack([self.zz, z, xdec1]))
            #pinjie = z + xdec1 + self.zz
            self.Z = pinjie.t().detach()

            y_pred = self.CN.clustering(self.Z)
        return y_pred,loss


    def run(self, eD =3,eF =3,e=50):

        self.DN.to(self.device)
        self.CN.to(self.device)
        self.SelfAttention.to(self.device)
        zclean, xdec1clean= self.DN.netclean(self.X)
        Z = zclean.t().detach()
        l = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward').fit(
            zclean.detach().cpu()).labels_
        centers = computeCentroids(zclean.detach().cpu().numpy(), l)
        self.CN.setC(torch.tensor(centers).t().to(self.device))
        self.CN._update_U(Z)
        optimizerCN = torch.optim.Adam(self.CN.parameters(), lr=self.lr)
        optimizerDN = torch.optim.Adam(self.DN.parameters(), lr=self.lr)
        train_loaderDN = DataLoader(TensorDataset(self.X), batch_size=self.batch_size, shuffle=True)

        for epoch in range(e):
            self.runDN(eD,train_loaderDN,optimizerDN)
            self.zz, reX = self.DN.netclean(self.X)
            self.Z=self.zz.t().detach()
            train_loaderCN= DataLoader(TensorDataset(self.zz.clone().detach().requires_grad_(True)), batch_size=self.batch_size,shuffle=True)
            y_pred,Loss =self.runCN(eF, train_loaderCN, optimizerCN)
            print('epoch-{}, loss={}'.format(epoch, Loss))
        return  y_pred



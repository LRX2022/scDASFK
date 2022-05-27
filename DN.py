from __future__ import print_function, division
import torch
import torch.nn as nn
class DN(nn.Module):
    def __init__(self,device="cuda",
                 dropouts=0.1,input =2500,output =512):
        super(DN, self).__init__()
        self.device=device
        self.ru = nn.LeakyReLU()  #
        self.drop = nn.Dropout(p=dropouts)

        self.enc1 = nn.Linear(input, output)
        self.enc2 = nn.Linear(output, output)
        self.dec1 = nn.Linear(output, input)
        self.MSELoss = nn.MSELoss()
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
    def netnoise(self, x):
        drop =x + (torch.rand(x.shape) * 1).to(device=self.device)
        enc1 = self.enc1(drop)
        enc1 = self.ru(enc1)
        enc2 = self.enc2(enc1)
        enc2 =self.drop(enc2)
        z = self.ru(enc2)
        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        return z, dec1
    def netclean(self, x):
        enc1 = self.enc1(x)
        enc1 = self.ru(enc1)
        enc2 = self.enc2(enc1)
        z = self.ru(enc2)
        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        return z, dec1

    def cosine(self, x):
        cosine = torch.pow(torch.sum(x ** 2.0, dim=1), 0.5)
        cosine = (x.t() / cosine).t()
        cosine = torch.mm(cosine, cosine.t())
        return cosine

    def _build_lossDN(self, x, zclean, znoise, xdec1noise, u):
        loss1 = self.MSELoss(xdec1noise, x)
        true_cosine = self.cosine(zclean)
        noise_cosine = self.cosine(u.to(self.device))
        loss3 = self.MSELoss(noise_cosine, true_cosine)
        noise_cosine = self.cosine(znoise)
        loss3 +=self.MSELoss(noise_cosine, true_cosine)
        return loss1+loss3



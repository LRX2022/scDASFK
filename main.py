import pandas as pd
import numpy as np
import torch
from preprocessing import reshapeX
from scDASFK import scDASFK
from utils import cal_clustering_metric

if __name__ == '__main__':

    la = pd.read_csv("dataset\kolo_lable.csv", header=0, index_col=0,sep=',')
    la = np.array(la).reshape(la.shape[0], )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = pd.read_csv("dataset\kolo_pre.csv", header=None,sep=',')
    X = np.array(X)
    print(X.shape)
    #X = reshapeX(X,k=3000)
    X = torch.FloatTensor(X).to(device)
    ii =6
    jj=4
    if(X.shape[0]>=1000):ii=3
    if (X.shape[0] >= 10000):ii = 1
    print(ii, jj)
    model = scDASFK(X=X, device =device,labels=la,input=X.shape[1],output=1000)  # len(np.unique(la)),output=nnnn
    y_pred = model.run(e=50,eD=ii,eF=jj)
    nmi, ari = cal_clustering_metric(la, y_pred)
    print("ARI:{},NMI:{}".format(ari, nmi))

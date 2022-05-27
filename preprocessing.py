
import scanpy as sc
import numpy as np
def reshapeX(data,k=3000):
    data = sc.AnnData(data)
    sc.pp.filter_genes(data, min_counts=int(data.shape[0] * 0.05))
    sc.pp.normalize_per_cell(data)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=k)
    data = data[:, data.var.highly_variable]
    data = np.array(data.X)
    print(data.shape)
    return data

def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


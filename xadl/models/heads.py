import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

class MahalanobisHead:
    def __init__(self):
        self.mean = None
        self.cov_inv = None
    def fit(self, Z):
        self.mean = Z.mean(axis=0)
        cov = np.cov(Z.T) + 1e-5*np.eye(Z.shape[1])
        self.cov_inv = np.linalg.inv(cov)
    def score(self, Z):
        d = Z - self.mean
        return np.einsum('bi,ij,bj->b', d, self.cov_inv, d)  # squared M-dist

class KNNHead:
    def __init__(self, k=10):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k)
        self.Z = None
    def fit(self, Z):
        self.Z = Z
        self.nn.fit(Z)
    def score(self, Z):
        dists, _ = self.nn.kneighbors(Z, n_neighbors=self.k, return_distance=True)
        return dists.mean(axis=1)

class OneClassSVMHead:
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        self.oc = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    def fit(self, Z):
        self.oc.fit(Z)
    def score(self, Z):
        return -self.oc.score_samples(Z)  # higher => more anomalous

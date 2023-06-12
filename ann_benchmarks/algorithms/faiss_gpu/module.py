import sys

# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import faiss
import numpy

from ..base.module import BaseANN

# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


class FaissGPU(BaseANN):
    def __init__(self, nlist):
        self.name = "FaissGPU(nlist={})".format(nlist)
        self._nlist = nlist
        self._res = faiss.StandardGpuResources()
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        self._index = faiss.GpuIndexIVFFlat(self._res, len(X[0]), self._nlist, faiss.METRIC_L2)
        self._index.train(X)
        self._index.add(X)

    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        v = v.astype(numpy.float32).reshape(1, -1)
        distances, labels = self._index.search(v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            if l != -1:
                r.append((l, d))
        return r

    def set_query_arguments(self, nprobe):
        self._index.nprobe = nprobe

    def batch_query(self, X, n):
        self.res = self._index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

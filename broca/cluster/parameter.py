import numpy as np
from sklearn.cluster import KMeans


def estimate_eps(dist_mat, n_closest=5):
    """
    Estimates possible eps values (to be used with DBSCAN)
    for a given distance matrix by looking at the largest distance "jumps"
    amongst the `n_closest` distances for each item.

    Tip: the value for `n_closest` is important - set it too large and you may only get
    really large distances which are uninformative. Set it too small and you may get
    premature cutoffs (i.e. select jumps which are really not that big).

    TO DO this could be fancier by calculating support for particular eps values,
    e.g. 80% are around 4.2 or w/e
    """
    dist_mat = dist_mat.copy()

    # To ignore i == j distances
    dist_mat[np.where(dist_mat == 0)] = np.inf
    estimates = []
    for i in range(dist_mat.shape[0]):
        # Indices of the n closest distances
        row = dist_mat[i]
        dists = sorted(np.partition(row, n_closest)[:n_closest])
        difs = [(x, y,
                (y - x)) for x, y in zip(dists, dists[1:])]
        eps_candidate, _, jump = max(difs, key=lambda x: x[2])

        estimates.append(eps_candidate)
    return sorted(estimates)


def estimate_k(X, max_k):
    """
    Estimate k for K-Means.

    Adapted from
    <https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/>
    """
    ks = range(1, max_k)
    fs = np.zeros(len(ks))

    # Special case K=1
    fs[0], Sk = _fK(1)

    # Rest of Ks
    for k in ks[1:]:
        fs[k-1], Sk = _fK(k, Skm1=Sk)
    return np.argmin(fs) + 1


def _fK(X, this_k, Skm1=0):
        Nd = X.shape[1]
        a = lambda k, Nd: 1 - 3./(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6.

        m = KMeans(n_clusters=this_k)
        labels = m.fit_predict(X)
        mu = m.cluster_centers_
        clusters = [[] for _ in range(max(labels) + 1)]
        for i, l in enumerate(labels):
            clusters[l].append(X[i])

        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(this_k) for c in clusters[i]])
        if this_k == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(this_k, Nd)*Skm1)
        return fs, Sk

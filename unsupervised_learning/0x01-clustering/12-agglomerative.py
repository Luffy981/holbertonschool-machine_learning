#!/usr/bin/env python3
"""Agglomerative clustering algorithm"""


import scipy.cluster.hierarchy as Hierarchy


def agglomerative(X, dist):
    """
    Agglomerative clustering algorithm.

    Args:
      X: data to cluster.
      dist: Maximum cophenetic distance for all clusters.

    Return:
      Point assignments for scatter plot
    """

    Z = Hierarchy.linkage(X, 'ward')
    scatt = Hierarchy.fcluster(Z, dist, 'distance')

    print(set(scatt))
    thresh = len(set(scatt))

    dn = Hierarchy.dendrogram(Z, color_threshold=dist, above_threshold_color='b')
    plt.show()
    plt.close()

    return scatt

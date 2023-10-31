#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
import pickle
import numpy as np
import networkx as nx
from multiprocessing import Pool
import pandas as pd
from lib.ci import collective_influence


def vg_ci(_input):
    file_, PI, T = _input
    graph_file = os.path.join('../VG_new_20', PI, file_)
    with open(graph_file, 'rb') as fp:
        vgs = pickle.load(fp)
    cis = {}
    for d, adj in vgs.items():
        labels = np.array([str(i) for i in range(20)])
        G = nx.Graph()
        for i in range(T):
            vg_adjs = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)
        cis[d] = collective_influence(G)
    ci_file = os.path.join('../CI', PI, '%s.json' % file_[:-7])
    with open(ci_file, 'w') as fp:
        json.dump(cis, fp)


if __name__ == '__main__':
    vol_price = ['close', 'vol', 'amount', 'high', 'open', 'low']

    weights = []
    for PI in vol_price:
        graph_file = os.path.join('../VG_new_20', PI, '000001.SZ.pickle')
        with open(graph_file, 'rb') as fp:
            vgs = pickle.load(fp)
        labels = np.array([str(i) for i in range(20)])
        G = nx.Graph()
        for i in range(20):
            vg_adjs = labels[np.where(vgs['20191227'][i] == 1)]
            edges = list(zip(*[[labels[i]] * len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)
        wlist = list(collective_influence(G).values())
        weights.append(wlist)


    for PI in vol_price:
        graph_file = os.path.join('../VG_new_20', PI, '000001.SZ.pickle')
        with open(graph_file, 'rb') as fp:
            vgs = pickle.load(fp)
        pd.DataFrame(vgs['20191227']).to_csv(f'{PI}_VG_mat.csv')

    T = 20
    for PI in vol_price:
        vg_dir = os.path.join('../VG_new_20/', PI)
        ci_dir = os.path.join('../CI', PI)
        if not os.path.exists(ci_dir):
            os.makedirs(ci_dir)
        # pool = Pool()
        # pool.map(vg_ci, [(f, PI, T) for f in os.listdir(vg_dir)])
        # pool.close()
        # pool.join()
        for f, PI, T in [(f, PI, T) for f in os.listdir(vg_dir)]:
            vg_ci((f, PI, T))


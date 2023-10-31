#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
import traceback
from tqdm import tqdm
from codes.utils.timedate_utils import TimeDateUtil
from codes.utils.logger_utils import LoggerUtils
from ge.models.struc2vec import Struc2Vec
curr_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
project_name = 'factor_framework'
project_name = curr_path[curr_path.find(project_name):].split('/')[0]
root_path = curr_path[:curr_path.find(project_name) + len(project_name)]
data_cache_path = root_path + '/data/cache_data'

PWD = os.path.dirname(os.path.realpath(__file__))
import time

def struc2vec_embedding(input_):
    file_, em_size, PI = input_
    vg_file = os.path.join('../VG', PI, file_)
    with open(vg_file, 'rb') as fp:
        vgs = pickle.load(fp)
    em_dir = os.path.join('../Struc2vec' , PI)
    if not os.path.exists(em_dir):
        os.makedirs(em_dir)
    ems = {}
    for d, adj in vgs.items():
        try:
            labels = np.array([str(i) for i in range(20)])
            G = nx.Graph()
            for i in range(len(labels)):
                adj_nodes = labels[np.where(adj[i] == 1)]
                edges = list(zip(*[[labels[i]]*len(adj_nodes), adj_nodes]))
                G.add_edges_from(edges)
            model = Struc2Vec(G, walk_length=10, num_walks=80, workers=40, verbose=40, stay_prob=0.3, opt1_reduce_len=True, opt2_reduce_sim_calc=True, opt3_num_layers=None, temp_path=f'./temp_struc2vec_{PI}_{d}/', reuse=False) #init model
            model.train(embed_size=em_size, window_size=3, workers=40, iter=5)  # train model
            embeddings = model.get_embeddings()  # get embedding vectors
            ems[d] = {k: v.tolist() for k, v in embeddings.items()}
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            LoggerUtils().get_logger().error(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            LoggerUtils().get_logger().error('[struc2vec] struc2vec error')
            continue
    with open(os.path.join(em_dir, '%s.json' % file_[:-7]), 'w') as fp:
        json.dump(ems, fp)
    return 1


def multi_process():
    '''并行计算, 算一个指标的embedding非常慢'''
    # 取前10天

    variable_list = ['open']
    LoggerUtils().get_logger().info('[PriceEmbedding] Started.')
    samples_data = []
    # cal = pd.read_pickle(data_cache_path + '/trade_cal.pkl')
    # cal = cal.query("is_trading_day==1")
    Dim = 32
    pool = multiprocessing.Pool(1)
    var = 'high'
    # for var in variable_list:
    vg_dir = os.path.join('../VG', var)
    for f in tqdm(os.listdir(vg_dir)):
        res = pool.apply_async(struc2vec_embedding, ((f, Dim, var),))
        samples_data.append(res)
    pool.close()
    pool.join()
    # # 2) 收集结果
    # data = []
    # for f in samples_data:
    #     data.append(f.get())
    # res = np.sum(np.array(samples_data))
    # print(f"res: {res}")
    return

if __name__ == '__main__':
    # t1 = time.time()
    multi_process()
    # cost = time.time() - t1
    # print("complete! time cost: ", cost)




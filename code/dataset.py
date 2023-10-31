#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import math
import json
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool
from functools import partial
from codes.utils.file_writer import FileWriter
sm = 1

PWD = os.path.dirname(os.path.realpath(__file__))
# all stocks
files = os.listdir('../data')[:5]
x_column = ['close', 'open', 'high', 'low', 'vol', 'amount']
y_column = 'close'
target = 1

class Dataset():
    def __init__(self, year, season):
        self.year = year
        self.season = season
        start_date = datetime(int(year), 1, 1)
        days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
        if season is None:
            self.days = [d for d in days if '%s0101' % year <= d <= '%s1231' % year][:5]
        else:
            sm, em = str((self.season - 1) * 3 + 1).zfill(2), str(self.season * 3).zfill(2)
            self.days = [d for d in days if '%s%s01' % (self.year, sm) <= d <= '%s%s31' % (self.year, em)][:5]
        self.global_ems = {f: {xc: self.load_embedding(f, xc, self.days) for xc in x_column} for f in files}
        self.global_df = {f: self.load_stock(f) for f in files}
        self.season = season
        self.global_ci = {f: {xc: self.load_ci(f, xc) for xc in x_column} for f in files}
        print(f"begin generating data for {year} {season}")

    def load_stock(self, s):
        df = pd.read_csv(os.path.join('../data', s), index_col=0)
        df.set_index(df.index.astype('str'), inplace=True)
        return df


    def load_ci(self, f, xi='close'):
        with open(os.path.join('../CI', xi, '%s.json' % f[:-4])) as fp:
            return json.load(fp)


    def load_embedding(self, f, xi='close', ti=None):
        with open(os.path.join('../Struc2vec', xi, '%s.json' % f[:-4])) as fp:
            j = json.load(fp)
        if ti is not None:
            return {d: j[d] for d in ti if d in j}
        return j


    def z_score(self, df):
        return (df - df.mean()) / df.std()


    def stock_sample(self, input_):
        print(f"processing {input_}")
        year = input_[-1][:4]
        input_ = input_
        s, d = input_
        T = 20
        df = self.global_df[s]
        if d not in df.index:
            return
        iloc = list(df.index).index(d) + 1
        if iloc < T:  # not enough history data
            return
        xss = {}
        for xi in x_column:
            # t
            t = 1 if df.iloc[iloc+target-1,:][xi] > df.loc[d, xi] else 0
            # y
            y = df.iloc[iloc-T:iloc][xi].copy()
            yz = np.array(self.z_score(y))
            if np.isnan(yz).any():
                return
            # ems
            ems = self.global_ems[s][xi]
            if d not in ems:
                return
            keys = ['%s' % i for i in range(T)]
            emd = np.array([ems[d][k] for k in keys])
            if len(emd) < T:
                return
            # ci
            cis = self.global_ci[s][xi]
            if d not in cis:
                return
            cid = cis[d]
            cid = [cid[str(i)] for i in range(T)]
            ciz = np.array(self.z_score(np.array(cid)))
            if np.isnan(ciz).any():
                ciz = np.array(cid)
            xss['%s_ems' % xi] = emd
            xss['%s_ys' % xi] = yz
            xss['%s_cis' % xi] = ciz
            xss['%s_t' % xi] = t
        res = [s, d, \
               xss['close_t'], xss['close_ems'], xss['close_ys'], xss['close_cis'], \
               xss['open_t'], xss['open_ems'], xss['open_ys'], xss['open_cis'], \
               xss['high_t'], xss['high_ems'], xss['high_ys'], xss['high_cis'], \
               xss['low_t'], xss['low_ems'], xss['low_ys'], xss['low_cis'], \
               xss['vol_t'], xss['vol_ems'], xss['vol_ys'], xss['vol_cis'], \
               xss['amount_t'], xss['amount_ems'], xss['amount_ys'], xss['amount_cis']]
        # FileWriter().object_dump_to_file(res, f'../dataset/tmp/{d}_{s[:-4]}.pkl')
        return res


    def sample_by_dates(self, dates):
        fds = [(f, d) for d in dates for f in files]
        pool = Pool(1)
        # pfunc = partial(stock_sample, fds)
        fds = fds
        samples = pool.map(self.stock_sample, fds)
        pool.close()
        pool.join()

        samples = filter(lambda s: s is not None, samples)
        stocks, days, \
        close_t, close_ems, close_ys, close_cis, \
        open_t, open_ems, open_ys, open_cis, \
        high_t, high_ems, high_ys, high_cis, \
        low_t, low_ems, low_ys, low_cis, \
        vol_t, vol_ems, vol_ys, vol_cis, \
        amount_t, amount_ems, amount_ys, amount_cis = zip(*samples)
        return {'stock': np.array(stocks), 'day': np.array(days),
                'close_t': np.array(close_t), 'close_ems': np.array(close_ems), 'close_ys': np.array(close_ys), 'close_cis': np.array(close_cis),
                'open_t': np.array(open_t), 'open_ems': np.array(open_ems), 'open_ys': np.array(open_ys), 'open_cis': np.array(open_cis),
                'high_t': np.array(high_t), 'high_ems': np.array(high_ems), 'high_ys': np.array(high_ys), 'high_cis': np.array(high_cis),
                'low_t': np.array(low_t), 'low_ems': np.array(low_ems), 'low_ys': np.array(low_ys), 'low_cis': np.array(low_cis),
                'vol_t': np.array(vol_t), 'vol_ems': np.array(vol_ems), 'vol_ys': np.array(vol_ys), 'vol_cis': np.array(vol_cis),
                'amount_t': np.array(amount_t), 'amount_ems': np.array(amount_ems), 'amount_ys': np.array(amount_ys), 'amount_cis': np.array(amount_cis),
                }


    def generate_data_year(self):
        # start_date = datetime(self.year, 1, 1)
        # days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
        # days = [d for d in days if '%s0101' % self.year <= d <= '%s1231' % self.year]
        dataset = self.sample_by_dates(self.days)
        with open(os.path.join('../dataset', '%s.pickle' % self.year), 'wb') as fp:
            pickle.dump(dataset, fp)


    def generate_data_season(self):
        dataset = self.sample_by_dates(self.days)
        with open(os.path.join('../dataset', '%s_S%s.pickle' % (self.year, self.season)), 'wb') as fp:
            pickle.dump(dataset, fp)


if __name__ == '__main__':
    # files = os.listdir('../data')
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')

    # import time
    # for y in range(2019, 2009, -1):
    #     t1 = time.time()
    #     print(y)
    #     Data = Dataset(year=y, season=None)
    #     Data.generate_data_year()
    #     print("time cost: ", time.time() - t1)

    for m in range(1, 5):
        print(m)
        Data = Dataset(year=2020, season=m)
        Data.generate_data_season()



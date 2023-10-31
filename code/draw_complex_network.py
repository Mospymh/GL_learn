# draw complex network
# coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd


def visibility_graph(series):
    visibility_graph_edges = []
    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    n = 1
    for magnitude in series:
        tseries.append((n, magnitude))
        n += 1

    for a, b in combinations(tseries, 2):
        # two points, maybe connect
        (ta, ya) = a  # 实现（1,2)（1,3）(1,4)(1,5)--(2,3)(2,4)(2,5)--(3,4)(3,5)--(4,5)任意两个边相互比较
        (tb, yb) = b
        connect = True
        medium = tseries[ta:tb - 1]  # 此处需要多留意，ta是1到k，而tseris是从0下标开始  所以此处不能不是[ta+1:tb]

        for tc, yc in medium:
            # print yc,(yb + (ya - yb) * ( float(tb - tc) / (tb - ta) ))#一定要float(tb-tc)/(tb-ta)因为计算机里1/2为0,1.0/2才为0.5
            if yc > yb + (ya - yb) * (float(tb - tc) / (tb - ta)):
                connect = False
        if connect:
            visibility_graph_edges.append((ta, tb))
    return visibility_graph_edges


def DrawVisibilityGraph(graphEdges):
    G = nx.Graph()
    # G = nx.DiGraph()# G=nx.Graph()无向图    G=nx.DiGraph()有向图
    # add the edge to graph G
    for i in graphEdges:
        G.add_edge(i[0], i[1])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, threshold=0.01)
    # nx.is_weighted(G, [0.1])
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=(0.75, 0, 0))
    nx.draw_networkx_edges(G, pos, width=3)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif', font_color='w')
    plt.savefig("test.png")   #save the picture
    return


if __name__ == "__main__":
    # time series
    var = 'open'
    sample = pd.read_csv('../data/000001.SZ.csv')
    for var in ['open', 'high', 'low', 'close', 'amount', 'vol']:
        series = sample[var][:20].values
        tmp = pd.DataFrame({'0': series})
        tmp.to_csv(f'{var}_network_plot.csv')
        # get the graph edges
        graphEdges = visibility_graph(series)
        # draw the visibility graph
        DrawVisibilityGraph(graphEdges)
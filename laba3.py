import pyrgg
import pandas as pd
import random
import networkx as nx
from queue import Queue
import matplotlib.pyplot as plt
import numpy as np

documents = dict()
k = 50

def new_document(v):
    num = random.randint(0, 2 ** 32)
    while num in documents.values():
        num = random.randint(0, 2 ** 32)
    documents[v] = num
    lis = addtolist([], d1_ids[v], num)
    used = set()
    while True:
        change = False
        for to in lis:
            if to not in used:
                used.add(to)
                lis = addtolist(lis, to, num)
                change = True
                break
        if not change:
            break
    if v not in files:
        files[v] = dict()
    files[v][num] = d1_ids[v]
    for id in lis:
        d_table[d2_ver[id]].add(d1_ids[v])
        if d2_ver[id] not in files:
            files[d2_ver[id]] = dict()
        files[d2_ver[id]][num] = d1_ids[v]

def gen_graph():
    pyrgg.graph_gen.csv_maker('test', 1, 1, 1000, 3, 10, False, False, False, False)
    df = pd.read_csv('test.csv', names=['from', 'to', 'w'])
    graph = [[] for i in range(1001)]
    for index, row in df.iterrows():
        graph[row['from']].append(row['to'])
        graph[row['to']].append(row['from'])
    return graph

graph = gen_graph()


def way(graph):
    mat_next = [[j if i == j else -1 for j in range(1001)] for i in range(1001)]
    mat_dist = [[0 if i == j else 1e9 for j in range(1001)] for i in range(1001)]
    for i in range(1, 1001):
        q = Queue()
        q.put(i)
        while not q.empty():
            fr = q.get()
            for to in graph[fr]:
                if mat_next[to][i] == -1:
                    mat_next[to][i] = fr
                    mat_dist[to][i] = mat_dist[fr][i] + 1
                    q.put(to)
    return mat_next, mat_dist

def cutlist(doc, lis):
    sorted(lis, key=lambda x: x ^ doc)
    lis = lis[:k]
    return lis

def addtolist(lis, id, doc):
    lis += list(d_table[d2_ver[id]])
    lis = cutlist(doc, lis)
    return lis

files = dict()


def table(graph):
    d1, d2 = dict(), dict()
    d_table = dict()
    for i in range(1, 1001):
        ch = random.randint(0, 2 ** 32)
        while ch in d2:
            ch = random.randint(0, 2 ** 32)
        d1[i] = ch
        d2[ch] = i
    for i in range(1, 1001):
        d_table[i] = set()
        d_table[i].add(d1[i])
        for j in graph[i]:
            d_table[i].add(d1[j])
    return d1, d2, d_table

d1_ids, d2_ver, d_table = table(graph)

def get_path(v, doc):
    lis = addtolist([], d1_ids[v], doc)
    for i in lis:
        flag, path = dfs(d2_ver[i], doc, {v})
        if flag:
            return [v] + path
    return []


def draw(graph, v, to, real_path):
    g = nx.Graph()
    colors = []
    for i in range(1, 1001):
        g.add_node(i)
        if i == v:
            colors.append('blue')
        elif i == to:
            colors.append('red')
        else:
            colors.append('black')
    pair_set = set()
    for i in range(0, len(real_path) - 1):
        g.add_edge(real_path[i], real_path[i + 1])
        pair_set.add((real_path[i], real_path[i + 1]))
        pair_set.add((real_path[i + 1], real_path[i]))
    for i in range(1, 1001):
        for j in graph[i]:
            if (i, j) not in pair_set:
                g.add_edge(i, j)
    colors_edge = []
    width = []
    for p in g.edges:
        if p in pair_set:
            colors_edge.append('green')
            width.append(2)
        else:
            colors_edge.append('grey')
            width.append(0.1)
    nx.draw_random(g, with_labels=False, node_size=25, node_color=colors, width=width, edge_color=colors_edge)
    plt.show()

def dfs(v, doc, used):
    used.add(v)
    if doc in files[v]:
        return True, [v, d2_ver[files[v][doc]]]
    lis = addtolist([], d1_ids[v], doc)
    for id in lis:
        if d2_ver[id] not in used:
            flag, path = dfs(d2_ver[id], doc, used)
            if flag:
                return flag, [v] + path
    return False, []

if __name__ == '__main__':
    mat_next, mat_dist = way(graph)
    for i in range(1, 1001):
        new_document(i)
    table_sizes = [len(d_table[i]) for i in range(1, 1001)]


    path_sizes = []
    for i in range(1000):
        v = random.randint(1, 1000)
        to = random.randint(1, 1000)
        while v == to:
            to = random.randint(1, 1000)
        doc = documents[to]
        path = get_path(v, doc)
        real_path = [v]
        for cur in path:
            while real_path[-1] != cur:
                real_path.append(mat_next[real_path[-1]][cur])
        path_sizes.append(len(real_path))

    #Min длина пути
    min(path_sizes)
    #Av длина пути
    np.mean(path_sizes)
    #Max длина пути
    max(path_sizes)
    draw(graph, v, to, real_path)
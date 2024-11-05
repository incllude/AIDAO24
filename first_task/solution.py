from nilearn.connectome import ConnectivityMeasure
from collections import Counter
from copy import deepcopy
from pathlib import Path
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import json


def get_connectome(timeseries, **kwargs):
    
    conn = ConnectivityMeasure(**kwargs).fit_transform(timeseries)
    if conn.shape[-2] == conn.shape[-1]:
        for i in range(conn.shape[0]):
            np.fill_diagonal(conn[i], 0)
    return conn


def calc_dist_matrix(from_ds, to_ds, dist_func, last_dim=(1,), fill_diag=True):
    
    n = from_ds.shape[0]
    m = to_ds.shape[0]
    dist_matrix = np.full((n, m, *last_dim), float("inf"))
    
    for i, x in enumerate(tqdm(from_ds)):
        for j, y in enumerate(to_ds):
            if (fill_diag and i == j) or (i > j and n == m):
                continue
            dist_matrix[i, j] = dist_func(x, y)
            dist_matrix[j, i] = dist_matrix[i, j].copy()
            
    return dist_matrix


def find_components_of_size(graph, size):
    
    result = []
    thresholds = sorted(set(weight for _, _, weight in graph.edges(data='weight')), reverse=True)
    prev_components = None
    per_cnt = 0
    
    for threshold in thresholds:
        binary_graph = nx.Graph()
        binary_graph.add_nodes_from(graph.nodes())
        
        for u, v, weight in graph.edges(data='weight'):
            if weight >= threshold:
                binary_graph.add_edge(u, v)
        
        components = list(nx.connected_components(binary_graph))
        
        for component in components:
            if len(component) > size:
                per_cnt += 1
                prev_component = [c for c in prev_components if set(c) & set(component)]
                f, s = sorted(prev_component, key=lambda x: len(x), reverse=True)
                f, s = list(f), list(s)
                s = np.random.choice(s, size=(size-len(f)), replace=False)
                s = s.tolist()
                component = f + s
            if len(component) == size:
                result.append((threshold, component))
                graph.remove_nodes_from(component)
                
        prev_components = deepcopy(components)
        
        if not graph.nodes():
            break
    
    return result, per_cnt


def group_by(dist_mat, top1, top2, verbose=False):
    
    n = dist_mat.shape[0]
    if top1 == -1:
        top1 = np.arange(1, n)
    similarity = np.sort(dist_mat, axis=-1)[:, ::-1][:, top1]
    siblings = np.argsort(dist_mat, axis=-1)[:, ::-1][:, top1]
    participants = []

    
    G = nx.Graph()
    for i in range(n):
        for j in range(siblings.shape[1]):
            G.add_edge(i, siblings[i, j], weight=similarity[i, j])

    components, per_cnt = find_components_of_size(G, size=8)
    if verbose:
        print(f"After first filter:  {len(list(G.nodes()))}")
    for threshold, component in components:
        participants.append(list(component))

    
    bad_idxs = np.array(list(G.nodes()))
    if bad_idxs.shape[0] == 0:
        return np.array(participants), per_cnt
    dist_mat_ = dist_mat[bad_idxs][:, bad_idxs]
    n = dist_mat_.shape[0]
    if top2 == -1:
        top2 = np.arange(1, n)
    similarity = np.sort(dist_mat_, axis=-1)[:, ::-1][:, top2]
    siblings = np.argsort(dist_mat_, axis=-1)[:, ::-1][:, top2]

    
    G = nx.Graph()
    for i in range(n):
        for j in range(siblings.shape[1]):
            G.add_edge(i, siblings[i, j], weight=similarity[i, j])

    components, per_cnt = find_components_of_size(G, size=8)
    if verbose:
        print(f"After second filter: {len(list(G.nodes()))}")
    for threshold, component in components:
        participants.append(bad_idxs[list(component)].tolist())

    return np.array(participants), per_cnt


data = np.load("./data/hb.npy")
bn_data = np.stack(list(filter(lambda x: ~np.isnan(x).any(), data)))[..., :210]

bn_data = (bn_data - bn_data.mean(axis=0, keepdims=True)) / bn_data.std(axis=0, keepdims=True)
bn_data_cm = get_connectome(bn_data,
                            kind="precision",
                            standardize=True,
                            discard_diagonal=True,
                            vectorize=False)


n, m, _ = bn_data_cm.shape
bn_data_cm_bw = np.empty((n, m, m // 2))

for i in range(m):
    if i % 2 == 0:
        bn_data_cm_bw[:, i, :] = bn_data_cm[:, i, 1::2]
    else:
        bn_data_cm_bw[:, i, :] = bn_data_cm[:, i, ::2]
        
bn_data_cm_bw_flattened = bn_data_cm_bw.reshape(n, -1)
dist_matrix = calc_dist_matrix(bn_data_cm_bw_flattened,
                               bn_data_cm_bw_flattened,
                               lambda x, y: np.round(np.corrcoef(x, y)[0, 1], 10))[..., 0]


bn_clusters, errors = group_by(dist_matrix, -1, -1)
assert errors == 0

bn_labels = np.empty(bn_data.shape[0])
sh_labels = np.empty(bn_data.shape[0])

for l, cluster in enumerate(bn_clusters):
    bn_labels[cluster] = l
    
with open("./data/bn_to_sch.json", "r") as f:
    bn_to_sh = json.load(f)
    
bn_to_sh_idxs = np.empty(len(bn_to_sh))
bn_to_sh_temp = {int(k): v[0][0] for k, v in bn_to_sh.items()}

for k, v in bn_to_sh_temp.items():
    bn_to_sh_idxs[k] = v
    
bn_to_sh_idxs = bn_to_sh_idxs.astype(int)

for i in range(bn_labels.shape[0]):
    sh_labels[bn_to_sh_idxs[i]] = bn_labels[i]

prediction = []
i, j = 0, 0 # i — brainnetome, j — schaefer

for a, x in enumerate(data):
    if np.isnan(x).any():
        prediction.append(int(sh_labels[j]))
        j += 1
    else:
        prediction.append(int(bn_labels[i]))
        i += 1
    
    
pd.DataFrame({'prediction': prediction}).to_csv('submission.csv', index=False)

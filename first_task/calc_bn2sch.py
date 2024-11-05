import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from numpy import correlate
import json
import os

if not os.path.exists('data_merge'):
    os.makedirs('data_merge')


def cross_correlation(ts1, ts2):
    ts1 = ts1 - np.mean(ts1)
    ts2 = ts2 - np.mean(ts2)

    correlation = correlate(ts1, ts2, mode='valid')
    correlation /= (np.std(ts1) * np.std(ts2) * len(ts1))

    return correlation[0]

def cross_correlation_matrix(sample_x, sample_y):
    num_i, num_j = sample_x.shape[1], sample_y.shape[1]
    corr_matrix = np.zeros((num_i, num_j))

    for i in range(num_i):
        for j in range(num_j):
            corr_matrix[i, j] = cross_correlation(sample_x[:, i], sample_y[:, j])

    return corr_matrix


def process_single_sample(sample_idx, bn_data, sh_data):
    sample_x = bn_data[sample_idx]
    results = []

    for sample_idy in range(sh_data.shape[0]):
        sample_y = sh_data[sample_idy]
        corr_matrix = cross_correlation_matrix(sample_x, sample_y)

        high_corr_count = np.sum(corr_matrix > 0.8)
        high_corr_mean = np.mean(corr_matrix[corr_matrix > 0.8]) if high_corr_count > 0 else 0
        combined_score = high_corr_count * high_corr_mean

        results.append(combined_score)

    top_5_indices = np.argsort(results)[-5:][::-1]
    top_5_scores = [results[idx] for idx in top_5_indices]
    kek = {int(sample_idx): [list(map(int, top_5_indices)), list(map(float, top_5_scores))]}
    with open(f'data_merge/merge_{sample_idx}.json', 'w') as f:
        json.dump(kek, f)
    return top_5_scores


def main():
    data = np.load('data/ihb.npy')
    bn_data = np.stack(list(filter(lambda x: ~np.isnan(x).any(), data)))[..., :210]
    sh_data = np.stack(list(filter(lambda x: np.isnan(x).any(), data)))[..., :200]

    num_bn_samples = bn_data.shape[0]

    worker = partial(process_single_sample, bn_data=bn_data, sh_data=sh_data)

    num_workers = cpu_count() - 1
    with Pool(processes=num_workers) as pool:
        cross_corr_features = list(tqdm(pool.imap(worker, range(num_bn_samples)), total=num_bn_samples))

if __name__ == '__main__':
    main()

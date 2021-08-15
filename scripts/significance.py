# need new version of scipy for 1-sided t-test
!pip install -U scipy

import os
import numpy as np
from tensorflow.python import pywrap_tensorflow

path = './tmp'  # this folder should include the folders like metrics_mslr_lambda_rank_loss
datasets = ['mslr', 'yahoo']
losses_base = ['pairwise_logistic_loss', 'lambda_rank_loss', 'softmax_loss', 'approx_ndcg_loss']
losses_ours = ['neuralsort_permutation_loss', 'pirank_simple_loss']
# skipping precision as it was not in the tables, add if necessary
# dirs = {'opa': '_OPAMetric', 'arp': '_ARPMetric', 'mrr': '_MRRMetric', 'ndcg': '_NDCGMetric', 'precision': '_PrecisionMetric'}
dirs = {'opa': '_OPAMetric', 'arp': '_ARPMetric', 'mrr': '_MRRMetric', 'ndcg': '_NDCGMetric'}
topns = {'arp': ['0'], 'mrr': ['0'], 'ndcg': ['1', '3', '5', '10', '15'], 'precision': ['1', '3', '5', '10', '15'], 'opa': ['0']}

data = {}
print("-- Reading data")
for dataset in datasets:
    data[dataset] = {}
    for loss in losses_base + losses_ours:
        data[dataset][loss] = {}
        for metric, mdir in dirs.items():
            data[dataset][loss][metric] = {}
            for topn in topns[metric]:
                path_ = os.path.join(path, f'metrics_{dataset}_{loss}/{mdir}/{topn}')
                subdirs = os.listdir(path_)
                assert len(subdirs) == 1
                path_ = os.path.join(path_, subdirs[0])
                fnames = os.listdir(path_)
                assert len(fnames) == 395
                values = None
                for fname in fnames:
                    reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(path_, fname))
                    values_ = reader.get_tensor('values')
                    values = values_ if values is None else np.vstack([values, values_])
                assert values.shape[0] == 6306
                print(dataset, loss, metric, topn, values.shape, values.mean())
                data[dataset][loss][metric][topn] = values.mean(axis=tuple(range(1, values.ndim)))

import scipy.stats
import pandas as pd
print("-- Checking significance")
tables = {}
pvalues = {}
for dataset in datasets:
    print('Looking at', dataset)
    table = pd.DataFrame()
    for metric, mdir in dirs.items():
        factor = -1 if metric == 'arp' else +1
        for topn in topns[metric]:
            smetric = metric if len(topns[metric]) == 1 else f'{metric}@{topn}'
            base_means = []
            for loss in losses_base:
                mean = data[dataset][loss][metric][topn].mean()
                base_means.append((factor * mean, loss))
                table.loc[loss, smetric] = mean
            best_base = sorted(base_means)[-1][1]
            for loss in losses_ours:
                values_base = data[dataset][best_base][metric][topn]
                values_ours = data[dataset][loss][metric][topn]
                _, pvalue = scipy.stats.ttest_rel(factor * values_base, factor * values_ours, alternative='less')
                pvalue = 1 if np.isnan(pvalue) else pvalue
                print(smetric, best_base, 'vs', loss, 'improvement =', (factor * (values_ours - values_base)).mean(), 'pvalue =', pvalue)
                mean = values_ours.mean()
                if pvalue < .001:
                    mean = str(mean) + '***'
                elif pvalue < .01:
                    mean = str(mean) + '**'
                elif pvalue < .05:
                    mean = str(mean) + '*'
                table.loc[loss, smetric] = mean
    tables[dataset] = table
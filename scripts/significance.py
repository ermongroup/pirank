# need new version of scipy for 1-sided t-test
!pip install -U scipy

import os
import numpy as np
from tensorflow.python.training import py_checkpoint_reader

path = '/content/drive/MyDrive/WORK/neuralsort-top/experiments/'  # this folder should include the folders like metrics_mslr_lambda_rank_loss
runs = {
    'mslr': {
        'base': {
            'tau_1': 'metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1',
            'tau_100': 'metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_100',
        },
        'ours': {
            'tau_10': 'metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10',
        }
    }
}
dirs = {'opa': 'OPAMetric', 'arp': 'ARPMetric', 'mrr': 'MRRMetric', 'ndcg': 'NDCGMetric'}
topns = {'arp': ['0'], 'mrr': ['None'], 'ndcg': ['1', '3', '5', '10'], 'precision': ['1', '3', '5', '10', '15'], 'opa': ['0']}

data = {}
print("-- Reading data")
for dataset, druns in runs.items():
    data[dataset] = {}
    for side in ('base', 'ours'):
        sruns = druns[side]
        data[dataset][side] = {}
        for run, rdir in sruns.items():
            data[dataset][side][run] = {}
            for metric, mdir in dirs.items():
                data[dataset][side][run][metric] = {}
                for topn in topns[metric]:
                    cpath = os.path.join(path, f'{rdir}/{mdir}/{topn}')
                    subdirs = os.listdir(cpath)
                    assert len(subdirs) == 1
                    cpath = os.path.join(cpath, sorted(subdirs)[-1])
                    fnames = os.listdir(cpath)
                    assert len(fnames) == 1
                    cpath = os.path.join(cpath, sorted(fnames)[-1])
                    reader = py_checkpoint_reader.CheckpointReader(cpath)
                    values = reader.get_tensor('values')
                    assert values.shape == (6306, 1)
                    print(run, metric, topn, values.shape, values.mean())
                    data[dataset][side][run][metric][topn] = values.mean(axis=tuple(range(1, values.ndim)))

import scipy.stats
import pandas as pd
print("-- Checking significance")
tables = {}
pvalues = {}
for dataset in data:
    print('Looking at', dataset)
    table = pd.DataFrame()
    for metric, mdir in dirs.items():
        factor = -1 if metric == 'arp' else +1
        for topn in topns[metric]:
            smetric = metric if len(topns[metric]) == 1 else f'{metric}@{topn}'
            base_means = []
            for run, rdata in data[dataset]['base'].items():
                mean = rdata[metric][topn].mean()
                base_means.append((factor * mean, run))
                table.loc[run, smetric] = mean
            best_base = sorted(base_means)[-1][1]
            values_base = data[dataset]['base'][best_base][metric][topn]
            for run, rdata in data[dataset]['ours'].items():
                values_ours = rdata[metric][topn]
                _, pvalue = scipy.stats.ttest_rel(factor * values_base, factor * values_ours, alternative='less')
                pvalue = 1 if np.isnan(pvalue) else pvalue
                print(smetric, best_base, 'vs', run, 'improvement =', (factor * (values_ours - values_base)).mean(), 'pvalue =', pvalue)
                mean = values_ours.mean()
                if pvalue < .001:
                    mean = str(mean) + '***'
                elif pvalue < .01:
                    mean = str(mean) + '**'
                elif pvalue < .05:
                    mean = str(mean) + '*'
                table.loc[run, smetric] = mean
    tables[dataset] = table
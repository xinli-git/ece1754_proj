import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi, autotvm

import sys
import argparse

import datetime

import pandas as pd
import pathlib
import json
import matplotlib.pyplot as plt


def plot_overall_results(niters, combos, batch_size_options, filters, indirs, outdir):
    fig, (ax0, ax1) = plt.subplots(2)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    width = 1
    positions = (0, 10, 20)
    for combo in combos + [('cudnn',)]:
        cudnn = 'cudnn' in combo
        label = 'cudnn' if cudnn else \
                        "cost model: {}, eps: {}".format(*combo)
        csv_f = "torch_conv.csv" if cudnn else \
                "{}_{}_{}.csv".format(combo[0], combo[1], 10000)
        overall_res = [indir / csv_f for indir in indirs]
        overall_res_pds = [pd.read_csv(r) for r in overall_res]

        overall_res_pd = overall_res_pds[0]
        for pdd in overall_res_pds[1:]:
            overall_res_pd['best latency'] += pdd['best latency']
        overall_res_pd['best latency'] /= len(indirs)

        overall_res_0 = overall_res_pd[overall_res_pd['conv layer id'] == 0]
        overall_res_1 = overall_res_pd[overall_res_pd['conv layer id'] == 1]
        ax0.bar(positions , overall_res_0['best latency'],
                label=label)
        ax1.bar(positions , overall_res_1['best latency'],
                label=label)

        positions = [p + width for p in positions]


    for idx, ax in enumerate((ax0, ax1)):
        batch_sizes = [1, 8, 64]
        ax.set_xticks([i + 2.5 for i in (0, 10, 20)])
        ax.set_xticklabels(map(str, batch_sizes))
        ax.set(xlabel = "batch size", ylabel="best latency (ms)")
        ax.legend(bbox_to_anchor=(1.1, 1.1))
        ax.set(title=["First layer resnet", "Last layer resnet"][idx])
        #ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.grid(axis='y')
    fig.tight_layout()
    fig.savefig(outdir / 'overall_results.png')

def plot_progression(niters, combos, batch_size_options, filters, indirs, outdir, x_axis='iters'):

    assert x_axis in ['iters', 'hours']
    batch_sizes = [1, 8, 64]
    cuda_csv = [indir / "torch_conv.csv" for indir in indirs]
    cuda_res = pd.read_csv(cuda_csv[0])
    for c in cuda_csv[1:]:
        cuda_res.iloc[:, 2] += pd.read_csv(c).iloc[:, 2]
    cuda_res.iloc[:, 2] /= len(indirs)
    ylims = {
            1 : [0.07, 0.35],
            8 : [.5, 0.6],
            64 : [5, 4],
            }

    steps = np.arange(50, 10001, 50)
    for batch_size in batch_sizes:
        for idx in (0, 1):
            fig = plt.figure()
            ax = fig.gca()
            fig.set_figheight(10)
            fig.set_figwidth(15)
            for combo in combos:
                label = "cost model: {}, eps: {}".format(*combo)
                csv_f = "conv2d_layer_{}_eps_{}_niters_{}_batch_{}_filter_id_{}_partial_mins.csv".format(combo[0], combo[1], 10000, batch_size, idx)
                overall_res = [indir / csv_f for indir in indirs]
                overall_res_pd = [pd.read_csv(o) for o in overall_res]

                if "XGBoost" in combo[0]:
                    linestyle = 'solid'
                else:
                    linestyle = 'dashdot'

                if x_axis == 'iters':
                    x = steps
                else:
                    time_csv_f = "conv2d_layer_{}_eps_{}_niters_{}_batch_{}_filter_id_{}_timestamps.csv".format(combo[0], combo[1], 10000, batch_size, idx)
                    timestamp = pd.read_csv(indirs[0] / time_csv_f)
                    x = (timestamp.iloc[steps-1, 1] - timestamp.iloc[0, 1]) / 3600
                ax.plot(x, sum([o.iloc[:, 1] for o in overall_res_pd]) / len(indirs), label=label, linestyle=linestyle)


            cuda_val = cuda_res.loc[(cuda_res['batch size'] == batch_size) & (cuda_res['conv layer id'] == idx)]
            cuda_val = float(cuda_val.iloc[0, 2])
            ax.axhline(y=cuda_val, color='b', linestyle='--', label="cuDNN")

            ax.set(xlabel = x_axis, ylabel="best latency so far(ms)")
            ax.legend(bbox_to_anchor=(1.1, 1.1))
            ax.set(title=["First layer resnet, Batch {}", "Last layer resnet, Batch{}"][idx]\
                                .format(batch_size))
            ylim = ylims[batch_size][idx]
            ax.set_ylim(bottom=0, top=ylim)
            #ax.set_yscale("log", base=2)
            ax.grid(axis='y')
            fig.tight_layout()
            fig.savefig(outdir / "batch_{}_filter_id_{}_partial_mins_{}.png".format(batch_size, idx, x_axis))


if __name__ == "__main__":

    niters = 10000
    combos = [('random', 0.5),
                ('XGBoost', 0.5),
                ('lstm', 0.5), ]

    batch_size_options = [1, 8, 64]

    filtre_size_opionts = [\
            (224, 224, 64, 3, 7, 7, (2, 2), (3, 3)),
            (7, 7, 512, 512, 3, 3, (1, 1), (1, 1)),
                            ]
    indirs = [pathlib.Path("./measurements2_gpu_1/"),
              pathlib.Path("./measurements2_gpu_2/"),
              pathlib.Path("./measurements2_gpu_3/")]

    outdir = pathlib.Path("./plots_gpu_3_with_lstm_full/")
    outdir.mkdir(exist_ok=True)

    plot_overall_results(niters, combos, batch_size_options, filters=2, indirs=indirs, outdir=outdir)
    plot_progression(niters, combos, batch_size_options, filters=2, indirs=indirs, outdir=outdir)
    plot_progression(niters, combos, batch_size_options, filters=2, indirs=indirs, outdir=outdir, x_axis='hours')




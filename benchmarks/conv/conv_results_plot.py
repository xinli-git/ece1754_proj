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

overall_results = {
        "batch size" : [],
        "conv layer id" : [],
        "cost model" : [],
        "search eps" : [],
        "best latency" : [],
        }


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def collect_partial_results(log_file, niters):

    configs = list()
    latencies = np.empty(niters)
    timestamps = np.empty(niters, dtype=np.long)

    restart_counter = 0

    with open(log_file) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            configs.append(obj['i'])

            prev_timestamp = timestamps[idx - 1] if idx > 0 else timestamps[idx]
            cur_timestamp = obj['r'][3] - restart_counter
            if (cur_timestamp - prev_timestamp) > 100: # if its been too long
                restart_counter += cur_timestamp - prev_timestamp
                print("restart encoutered at {}, duration: {}".format(idx, cur_timestamp - prev_timestamp))
                cur_timestamp -= cur_timestamp - prev_timestamp

            timestamps[idx] = cur_timestamp

            errno = obj['r'][1]
            if errno != 0:
                latency = np.Inf
            else:
                latency = obj['r'][0][0] * 1000 # ms
            latencies[idx] = latency

    min_latency = np.min(latencies)
    partial_mins = []
    for idx in range(10, niters):
        partial_mins.append(np.min(latencies[:idx]))
    #fig = plt.figure(figsize=(32, 4.8 ))
    #ax = fig.add_subplot(111)
    #ax.plot(partial_mins)
    return pd.DataFrame(latencies), pd.DataFrame(timestamps), pd.DataFrame(partial_mins), min_latency

def measure_task(func, cost_model, search_eps, niters, prefix, args, outdir):

    t_name = func.__name__ + prefix

    target = tvm.target.Target("cuda")
    dev = tvm.cuda()

    N, H, W, CO, CI, KH, KW, strides, padding = args
    task = auto_scheduler.SearchTask(func=func, args=args, target=target)

    log_file = os.path.join("./results/", "{}.json".format(t_name))
    assert os.path.exists(log_file)

    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    tvm_func = tvm.build(sch, args, target)

    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH,
                                        KW)).astype(np.float32)

    data_tvm = tvm.nd.empty((N, CI, H, W), device=dev)
    weight_tvm = tvm.nd.empty((CO, CI, KH, KW), device=dev)
    out_tvm = tvm.nd.empty((N, CO, H / strides[0], W / strides[1]), device=dev)

    tvm_func(data_tvm, weight_tvm, out_tvm)

    # Evaluate execution time
    evaluator = tvm_func.time_evaluator(tvm_func.entry_name,
                                        dev,
                                        min_repeat_ms=500)
    res = np.mean(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000
    #plt.clf()
    latencies, timestamps, partial_mins, min_latency = \
            collect_partial_results(log_file, niters)
    #plt.savefig(outdir / "{}.png".format(t_name))


    latencies_file = outdir / "{}_latencies.csv".format(t_name)
    timestamps_file = outdir / "{}_timestamps.csv".format(t_name)
    partial_mins_file = outdir / "{}_partial_mins.csv".format(t_name)

    latencies.to_csv(latencies_file)
    timestamps.to_csv(timestamps_file)
    partial_mins.to_csv(partial_mins_file)

    return res, min_latency



if __name__ == "__main__":
    parser = argparse.ArgumentParser("benchmarking conv")
    parser.add_argument("niters", type=int)
    parser.add_argument("--cost_model", type=str, default="XGBoost", choices=["random", "XGBoost"])
    parser.add_argument("--search_eps", type=float, default=0.05)
    args = parser.parse_args()

    niters = args.niters
    cost_model = args.cost_model
    search_eps = args.search_eps

    batch_size_options = [1, 8, 64]

    filtre_size_opionts = [\
            (224, 224, 64, 3, 7, 7, (2, 2), (3, 3)),
            (7, 7, 512, 512, 3, 3, (1, 1), (1, 1)),
                            ]
    outdir = pathlib.Path("./measurements/")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / "{}_{}_{}.csv".format(cost_model, search_eps, niters)

    for idx, filter_size in enumerate(filtre_size_opionts):
        for batch_size in batch_size_options:

            args = (batch_size, ) + filter_size
            prefix = "_{}_eps_{}_niters_{}_batch_{}_filter_id_{}"\
                        .format(cost_model, search_eps, niters, batch_size, idx)

            res, log_res = measure_task(conv2d_layer, cost_model, search_eps, niters, prefix, args, outdir)
            print(prefix, ":", "{:.3f} and {:.3f}".format(res, log_res))
            overall_results["batch size"].append(batch_size)
            overall_results["conv layer id"].append(idx)
            overall_results["cost model"].append(cost_model)
            overall_results["search eps"].append(search_eps)
            overall_results["best latency"].append(res)

    df = pd.DataFrame(data=overall_results)
    df.to_csv(outfile)



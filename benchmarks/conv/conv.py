import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi, autotvm

import sys
import argparse

import datetime


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def tune_task(func, cost_model, search_eps, niters, prefix, args):

    t_name = func.__name__ + prefix

    print("Tuning {}".format(t_name))

    target = tvm.target.Target("cuda")

    N, H, W, CO, CI, KH, KW, strides, padding = args

    task = auto_scheduler.SearchTask(func=func, args=args, target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = os.path.join("./results_gpu_3/", "{}.json".format(t_name))

    PARAMS = {
        "eps_greedy": search_eps,
        "retry_search_one_round_on_empty": 1,
        "sample_init_min_population": 50,
        "sample_init_use_measured_ratio": 0.2,
        "evolutionary_search_population": 2048,
        "evolutionary_search_num_iters": 4,
        "evolutionary_search_mutation_prob": 0.85,
        "cpu_multi_level_tiling_structure": "SSRSRS",
        "gpu_multi_level_tiling_structure": "SSSRRSRS",
        # Notice: the default thread bind policy of GPU assumes the tiling structure to have at
        # least 3 spatial tiling levels in outermost
        "max_innermost_split_factor": 64,
        "max_vectorize_size": 16,
        "disable_change_compute_location": 0,
    }

    cost_model = auto_scheduler.XGBModel(adapative_training=True) \
            if cost_model == "XGBoost" else auto_scheduler.RandomModel()


    if not os.path.exists(log_file):
        search_policy = auto_scheduler.SketchPolicy(task,
                                                    cost_model,
                                                    params=PARAMS)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=niters,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

        task.tune(tune_option, search_policy=search_policy)

    else:
        print("Resuming with log file", log_file)
        with open(log_file) as f:
            num_trials_done = sum(1 for _ in f)
        num_trials_left = niters - num_trials_done
        print("Executed: {} trils, resuming {}".format(num_trials_done, num_trials_left))
        if num_trials_left > 0:
            if isinstance(cost_model, auto_scheduler.XGBModel):
                cost_model.update_from_file(log_file)
            search_policy = auto_scheduler.SketchPolicy(
                task, cost_model, params=PARAMS,
                init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
            )

            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=num_trials_left,
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                verbose=2,
            )

            task.tune(tune_option, search_policy=search_policy)
    print("Done!", datetime.datetime.now())

    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    #print("Lowered TIR:")
    #print(tvm.lower(sch, args, simple_mode=True))

    tvm_func = tvm.build(sch, args, target)

    # Check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH,
                                        KW)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.empty((N, CI, H, W), device=dev)
    weight_tvm = tvm.nd.empty((CO, CI, KH, KW), device=dev)
    out_tvm = tvm.nd.empty((N, CO, H / strides[0], W / strides[1]), device=dev)

    tvm_func(data_tvm, weight_tvm, out_tvm)

    # Evaluate execution time
    evaluator = tvm_func.time_evaluator(tvm_func.entry_name,
                                        dev,
                                        min_repeat_ms=500)
    print("Execution time of this operator: %.3f ms\n\n" %
          (np.mean(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000))

    #print("Equivalent python schedule:")
    #print(task.print_best(log_file, print_mode="schedule"))

    #print("CUDA source code:")
    #print(task.print_best(log_file, print_mode="cuda"))


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
    for batch_size in batch_size_options:

        for idx, filter_size in enumerate(filtre_size_opionts):
            args = (batch_size, ) + filter_size
            prefix = "_{}_eps_{}_niters_{}_batch_{}_filter_id_{}"\
                        .format(cost_model, search_eps, niters, batch_size, idx)
            tune_task(conv2d_layer, cost_model, search_eps, niters, prefix,
                      args)

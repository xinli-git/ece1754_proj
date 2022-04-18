import json
import os
import argparse
import tvm
from tvm.auto_scheduler  import ComputeDAG
from conv import conv2d_layer
import pathlib
from multiprocessing import Pool, Queue
from functools import partial
from tqdm import tqdm
import numpy as np
import pickle
from functools import reduce
import matplotlib.pyplot as plt

from training import train

noop = lambda x: np.zeros(2)
NUM_MAX_STAGES = 10
NUM_MAX_LOOPS = 50
NUM_ANNOTATION_TYPES = 12
NUM_TRANSFORM_TYPES = 14

MEMORY_SCOPE_DECODE = {
        'local' : 0,
        'shared' : 1,
        }

results_outdir = None

def extract_annotation(data):

    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    iteration_onehot = np.zeros(NUM_MAX_LOOPS)
    annotype_onehot = np.zeros(NUM_MAX_STAGES)

    transform_id = 0
    stage, iteration, anno_type = data[1:]

    transform_onehot[transform_id] = 1
    stage_onehot[stage] = 1.
    iteration_onehot[iteration] = 1.
    annotype_onehot[anno_type] = 1.

    return np.concatenate([transform_onehot, stage_onehot, iteration_onehot, annotype_onehot])


def extract_fuse(data):

    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)

    transform_id = 1
    stage, iterations = data[1:]

    transform_onehot[transform_id] = 1
    stage_onehot[stage] = 1.
    iter_embedding = []
    for iteration in iterations:
        iterations_onehot = np.zeros(NUM_MAX_LOOPS)
        iterations_onehot[iteration] = 1.
        iter_embedding.append(iterations_onehot)

    return np.concatenate([transform_onehot, stage_onehot, ] + iter_embedding)

def extract_pragma(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    iteration_onehot = np.zeros(NUM_MAX_LOOPS)

    transform_id = 2
    stage, iteration, pragma_val = data[1:]
    assert pragma_val.startswith('auto_unroll')

    transform_onehot[transform_id] = 1
    stage_onehot[stage] = 1.
    iteration_onehot[iteration] = 1.

    return np.concatenate([transform_onehot, stage_onehot, iteration_onehot])

def extract_reorder(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)

    transform_id = 3
    stage, new_order = data[1:]

    transform_onehot[transform_id] = 1
    stage_onehot[stage] = 1.
    new_order = np.array(new_order)

    return np.concatenate([transform_onehot, stage_onehot, new_order])


def extract_split(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    iteration_onehot = np.zeros(NUM_MAX_LOOPS)

    transform_id = 4
    stage, iteration, extend, factors, outer_to_inner = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    iteration_onehot[iteration] = 1.
    extend = np.array([extend])
    outer_to_inner = np.array([outer_to_inner])
    factors = np.array(factors)

    return np.concatenate([transform_onehot, stage_onehot, iteration_onehot, extend, outer_to_inner, factors])

def extract_follow_split(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    iteration_onehot = np.zeros(NUM_MAX_LOOPS)
    src_step_onehot = np.zeros(NUM_MAX_STAGES)
    level_onehot = np.zeros(4)

    transform_id = 5
    stage, iteration, src_step, levels = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    iteration_onehot[iteration] = 1.
    src_step_onehot[src_step] = 1.
    level_onehot[levels] = 1.

    return np.concatenate([transform_onehot, stage_onehot, iteration_onehot, src_step_onehot, level_onehot])

def extract_follow_fused_split(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    iteration_onehot = np.zeros(NUM_MAX_LOOPS)
    level_onehot = np.zeros(4)

    transform_id = 6
    stage, iteration, src_steps, levels, factor_or_nparts = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    iteration_onehot[iteration] = 1.

    steps_onehot = []
    for step in src_steps:
        src_step_onehot = np.zeros(NUM_MAX_STAGES)
        src_step_onehot[step] = 1.
        steps_onehot.append(src_step_onehot)
    steps_onehot = np.concatenate(steps_onehot)

    level_onehot[levels] = 1.
    factor_or_nparts = np.array([factor_or_nparts])

    return np.concatenate([transform_onehot, stage_onehot, iteration_onehot, steps_onehot, level_onehot, factor_or_nparts])

def extract_compute_at(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    target_stage_onehot = np.zeros(NUM_MAX_STAGES)
    target_iteration_onehot = np.zeros(NUM_MAX_LOOPS)

    transform_id = 7
    stage, target_stage, target_iteration = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    target_stage_onehot[target_stage] = 1.
    target_iteration_onehot[target_iteration] = 1.

    return np.concatenate([transform_onehot, stage_onehot, target_stage_onehot, target_iteration_onehot])

def extract_compute_inline(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)

    transform_id = 8
    stage = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.

    return np.concatenate([transform_onehot, stage_onehot, ])

def extract_cache_read(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    scope_onehot = np.zeros(len(MEMORY_SCOPE_DECODE))

    transform_id = 9
    stage, scope, readers = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    scope_onehot[MEMORY_SCOPE_DECODE[scope]] = 1.

    readers_onehot = []
    for reader in readers:
        reader_onehot = np.zeros(NUM_MAX_STAGES)
        reader_onehot[reader] = 1.
        readers_onehot.append(reader_onehot)

    readers_onehot = np.concatenate(readers_onehot)

    return  np.concatenate([transform_onehot, stage_onehot, scope_onehot, readers_onehot])

def extract_cache_write(data):
    transform_onehot = np.zeros(NUM_TRANSFORM_TYPES)
    stage_onehot = np.zeros(NUM_MAX_STAGES)
    scope_onehot = np.zeros(len(MEMORY_SCOPE_DECODE))

    transform_id = 10
    stage, scope = data[1:]

    transform_onehot[transform_id] = 1.
    stage_onehot[stage] = 1.
    scope_onehot[MEMORY_SCOPE_DECODE[scope]] = 1.
    return np.concatenate([transform_onehot, stage_onehot, scope_onehot])


class TunedSchedule:

    ExtractFrom = {
        "AN" : extract_annotation,    #"auto_scheduler.AnnotationStep"
        "FU" : extract_fuse,    #"auto_scheduler.FuseStep"
        "PR" : extract_pragma,    #"auto_scheduler.PragmaStep"
        "RE" : extract_reorder,    #"auto_scheduler.ReorderStep"
        "SP" : extract_split,    #"auto_scheduler.SplitStep"
        "FSP" : extract_follow_split,   #"auto_scheduler.FollowSplitStep"
        "FFSP" : extract_follow_fused_split,  #"auto_scheduler.FollowFusedSplitStep"
        "SA" : (lambda x: NotImplementedError("Not needed") ),    #"auto_scheduler.StorageAlignStep"
        "CA" : extract_compute_at,    #"auto_scheduler.ComputeAtStep"
        "CI" : extract_compute_inline,    #"auto_scheduler.ComputeInlineStep"
        "CR" : (lambda x: NotImplementedError("Not needed") ),    #"auto_scheduler.ComputeRootStep"
        "CHR" : extract_cache_read,   #"auto_scheduler.CacheReadStep"
        "CHW" : extract_cache_write,   #"auto_scheduler.CacheWriteStep"
        "RF" : (lambda x: NotImplementedError("Not needed") ),    # Rfactor
   }

    def __init__(self, inp, measure, inf, line):

#         for_lower = self.dag.apply_steps_from_state(inp.state)
#         assert len(inp.state.transform_steps) == 37
#
#         self.schedule, self.args = for_lower
#         self.func = tvm.lower(self.schedule, self.args)

        assert len(measure) == 1
        self.metadata, transforms = inp

        self.workload_key = self.metadata[0]
        self.dag = ComputeDAG(self.workload_key)

        self.transforms = transforms[1]
        if len(self.transforms) != 37:
            raise ValueError("At {}:{}\nExpect number of transforms to be 37, got {}\n{}"
                    .format(inf, line, len(self.transforms), self.transforms))

        self.features = self.extract_features()
        self.performance = np.array(measure)

    def numpy(self):
        return self.features, self.performance

    def extract_features(self):
        #global global_order
        #if global_order is None:
        #    global_order = tuple(t[0] for t in self.transforms)
        #else:
        #    xxx = tuple(t[0] for t in self.transforms)
        #    if global_order != xxx:
        #        print(global_order, xxx)
        features = []
        for transform in self.transforms:
            transform_name = transform[0]
            feature = TunedSchedule.ExtractFrom[transform_name](transform)
            features.append(feature)

        return features

    def get_similarity(self, sche, geomean=True):
        sims = self.get_similarity_by_features(sche)
        if geomean:
            return reduce(lambda x, y: x*y, sims)**(1.0/len(sims))
        else:
            return sum(sims) / len(sims)

    def get_similarity_by_features(self, sche):
        sims = []
        for a, b in zip(self.features, sche.features):
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
            sims.append(cos_sim)

        return sims


    def __str__(self):
        lines = ["Schedule for {}".format(self.workload_key)]
        lines.append("Performance: {:.4f}".format(float(self.performance)))
        for feature in self.features:
            lines.append('\t' + ''.join([str(int(f)) for f in feature]))

        return "\n".join(lines)

def process_schedule_file(grouped_files):
    global results_outdir
    group_schedules = []
    seps = []
    for ffs in grouped_files[1]:
        infile, outfile = ffs
        outpickle = outfile + ".pkl"
        if os.path.exists(outpickle):
             with open(outpickle, 'rb') as f:
                schedules = pickle.load(f)
        else:
            schedules = []
            with open(infile) as f:
                for idx, line in tqdm(enumerate(f), total=10000):
                    record = json.loads(line)
                    error_no = record['r'][1]
                    if error_no != 0:
                        continue

                    sche = TunedSchedule(record['i'], record['r'][0], infile, idx)
                    #print(len(sche.features), [f.shape for f in sche.features])
                    schedules.append(sche)


            with open(outpickle, 'wb') as f:
                pickle.dump(schedules, f, pickle.HIGHEST_PROTOCOL)

        seps.append((   infile,
                        len(group_schedules),
                        len(group_schedules) + len(schedules)
                    ))
        group_schedules += schedules
    plot_cosine(group_schedules, grouped_files[0], results_outdir, seps)

    train(group_schedules, grouped_files[0], results_outdir)

    return grouped_files[0], len(group_schedules)

def plot_cosine(schedules, group, outdir, seps):
    #print(best)
    #for s in sorted_schedules[:10]:
    #    print("Best", best.get_similarity(s))
    #for s in sorted_schedules[-10:]:
    #    print("Worst", best.get_similarity(s))

    first = min(schedules, key=lambda x: x.performance)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()

    for label, start, end in seps:
        label = label[label.find('conv2d') : label.find('_niters')]
        schedule_portion = schedules[start:end]
        perf_diffs = [sche.performance / first.performance for sche in schedule_portion]
        sche_diffs = [first.get_similarity(sche) for sche in schedule_portion]
        ax.scatter(sche_diffs, perf_diffs, label=label, alpha=0.5, s=8)
    ax.set(ylabel='Performance from first', xlabel="Cosine similarity from first")
    ax.set(title=group)
    ax.set(ylim=(1, 64))
    ax.grid(axis='y')
    ax.set_yscale("log", base=2)
    fig.legend()
    fig.tight_layout()
    fig.savefig(outdir / (group + '_cosine.png'))
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("benchmarking conv")
    parser.add_argument("indir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--exist_ok", action='store_true', default=False)

    args = parser.parse_args()

    indir = pathlib.Path(args.indir).resolve()
    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(exist_ok=args.exist_ok)
    results_outdir = outdir

    files = [(str(indir / f), str(outdir/f.name)[:-5] )  for f in indir.iterdir()]

    groups = ['batch_{}_filter_id_{}'.format(b, i) for b in (1, 8, 64) for i in (0, 1)]
    grouped_files = {k : [] for k in groups}
    for f in files:
        for group in groups:
            if group in str(f):
                grouped_files[group].append(f)

    #with Pool(8) as p:
    r = map(process_schedule_file, grouped_files.items())

    all_schedules = {}
    for f, schedules in tqdm(r, leave=False, total=len(grouped_files)):
        all_schedules[f] = schedules

    for f, a in all_schedules.items():
        print(f, a)


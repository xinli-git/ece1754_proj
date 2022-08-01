import tvm

import ctypes
import numpy as np
import tvm.auto_scheduler._ffi_api as _ffi_api
import tvm.auto_scheduler
from tvm.auto_scheduler.cost_model.cost_model import PythonBasedModel

import torch
from schedule_reader import TunedSchedule
import json
from training import ConvPerfPredictor

import torch.nn.functional as F

def prepare_feature_batch(schedules, ):
    batched_features = []

    f_maxlen = max([len(f) for f in schedules[0].features])
    for schedule in schedules:
        features = schedule.features
        features_padded = [F.pad(torch.tensor(f, dtype=torch.float, device='cuda:1'),
                                 (0, f_maxlen - len(f), )) \
                                            for f in features]
        features_padded = torch.cat([f.unsqueeze(0) for f in features_padded], axis=0)
        batched_features.append(features_padded)

    return torch.stack(batched_features)

#@tvm._ffi.register_object("auto_scheduler.LSTMModel")
class LSTMModel(PythonBasedModel):
    """A model that returns random estimation for all inputs"""

    def __init__(self, checkpoint=None, warmup_step=0,
                re_buffer_size=128, single_batch_iter=1024):
        super().__init__()
        self.checkpoint = checkpoint
        if checkpoint is not None:
            self.model = torch.load(checkpoint)
        else:
            hidden_dim = 64
            num_layers = 4
            self.model = ConvPerfPredictor(224, hidden_dim, num_layers)
        self.model.cuda(1)
        print("LSTMModel: Model initialized")
        print(self.model)
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.AdamW(self.model.parameters(),)
        self.update_step = 0
        self.warmup_step = warmup_step

        self.buffer = []
        self.re_buffer_size = re_buffer_size
        self.single_batch_iter = single_batch_iter

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).
        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        if self.checkpoint is not None:
            return
        self.model.train()

        inps_json = [json.loads(inp.serialize()[0]) for inp in inputs]
        inps_schedule = [TunedSchedule(inp) for inp in inps_json]
        batch = prepare_feature_batch(inps_schedule)
        batch = batch.cuda(1)
        actual = torch.tensor([float(r.costs[0]) if r.error_no ==0 else 10000. \
                                    for r in results ])
        actual = actual.cuda(1)
        truth = 1./actual
        batch_mse = 0
        #for idx, item in enumerate(batch):
        #    item = item.unsqueeze(0)
        for multi_iter in range(self.single_batch_iter):
            pred = self.model(batch).flatten()
            mse = self.loss(pred, truth)
            print("pred:  {}".format(pred))
            print("truth: {}".format(truth))
            mse.backward()
            self.optim.step()
            self.optim.zero_grad()
            batch_mse += float(mse)
        print("TRAINING: ", self.update_step, batch_mse / batch.size(0))
        self.update_step += len(batch)

    def predict(self, search_task, states):
        """Predict the scores of states
        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states
        Returns
        -------
        scores: List[float]
            The predicted scores for all states
        """
        if self.update_step < self.warmup_step:
            return [0.0 for _ in range(len(states))]
        self.model.eval()

        inps = [tvm.auto_scheduler.MeasureInput(search_task, s) for s in states]
        inps_json = [json.loads(inp.serialize()[0]) for inp in inps]
        inps_schedule = [TunedSchedule(inp) for inp in inps_json]
        batch = prepare_feature_batch(inps_schedule)
        batch.cuda(1)
        ret = self.model(batch)
        return [float(i) for i in ret]

import tvm

import ctypes
import numpy as np
import tvm.auto_scheduler._ffi_api as _ffi_api
import tvm.auto_scheduler
from tvm.auto_scheduler.cost_model.cost_model import PythonBasedModel

import torch
import schedule_reader
import json

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

    def __init__(self, checkpoint):
        super().__init__()
        self.model = torch.load(checkpoint).cuda(1)
        self.model.eval()
        print("LSTMModel: Model initialized")
        print(self.model)


    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).
        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        pass

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

        inps = [tvm.auto_scheduler.MeasureInput(search_task, s) for s in states]
        inps_json = [json.loads(inp.serialize()[0]) for inp in inps]
        inps_schedule = [schedule_reader.TunedSchedule(inp) for inp in inps_json]
        batch = prepare_feature_batch(inps_schedule)
        batch.cuda(1)
        ret = self.model(batch)
        return [float(i) for i in ret]

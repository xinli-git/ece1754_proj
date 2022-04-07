import json
from tvm.auto_scheduler import RecordReader
from conv import conv2d_layer

class ScheduleReader:
    def __init__(self, infile,
            workload_key,
            op_name,
            op_id,
            batch_size,
            cost_model,
            search_eps):
        self.infile = infile
        self.op_name = op_name
        self.op_id = op_id
        self.batch_size = batch_size
        self.cost_model = cost_model
        self.search_eps = search_eps

        self.records = tvm.auto_scheduler.load_records(infile)
        self.dag = tvm.auto_scheduler.ComputeDAG(workload_key)
    def



class TunedSchedule:
    @staticmethod
    def _parse_schedule(cls, params):
        pass


    def __init__(self, schedule_s):

        schedule = json.loads(schedule_s)
        schedule_params = TunedSchedule._parse_schedule(schedule['i'])




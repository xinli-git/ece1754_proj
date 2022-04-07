import json
from tvm.auto_scheduler import RecordReader


class ScheduleReader:
    def __init__(self, infile,
            op_name,
            od_id,
            batch_size,
            cost_model,
            search_eps):
        self.infile = infile
        self.op_name = op_name
        self.op_id = op_id
        self.batch_size = batch_size
        self.cost_model = cost_model
        self.search_eps = search_eps


        records = tvm.auto_scheduler.load_records(infile)
        # inp.task.hardware_params
        # inp.transp



class TunedSchedule:
    @staticmethod
    def _parse_schedule(cls, params):


    def __init__(self, schedule_s):

        schedule = json.loads(schedule_s)
        schedule_params = TunedSchedule._parse_schedule(schedule['i'])




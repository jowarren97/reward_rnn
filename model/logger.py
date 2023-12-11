from collections.abc import Iterable
from tensorboardX import SummaryWriter
import os

# /Volumes/jwarren/notebooks_paper/model/summaries

class LearningLogger:
    def __init__(self, objs=(), path=''):
        self.pointers = []
        self.log = dict()
        self.root = os.getcwd()
        self.profile_path = os.path.join(self.root, 'summaries', path)
        self.writer = SummaryWriter(self.profile_path)

        self.add_pointer(objs)

    def add_pointer(self, objs):
        if not is_iterable(objs):
            objs = (objs, )

        for obj in objs:
            if not hasattr(obj, 'get_log'):
                raise AttributeError(f'Class {type(obj)} has no attribute "get_log()"')
            else:
                self.pointers.append(obj)

    def get_logs(self):
        for obj in self.pointers:
            name = obj.__class__.__module__
            log = obj.get_log()
            self.log[name] = log

        return self.log
    
    def reset_logs(self):
        for obj in self.pointers:
            obj.reset_log()
    
    def to_tensorboard(self, epoch):
        for obj_name, obj_log in self.log.items():
            for var_name, var in obj_log.items():
                if var["tb"]:
                    if len(var["data"]):
                        data = var["data"][-1]
                        tb_path = '/'.join([obj_name, var_name])
                        self.writer.add_scalar(tb_path, data, epoch)

def is_iterable(obj):
    return isinstance(obj, Iterable)
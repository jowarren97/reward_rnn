from collections.abc import Iterable
from tensorboardX import SummaryWriter
import subprocess
from datetime import datetime
import os

class LearningLogger:
    def __init__(self, objs=()):
        self.pointers = []
        self.log = dict()
        self.root = os.getcwd()
        self.profile_path = os.path.join(self.root, 'summaries', get_current_date() + '_' + get_git_commit_id())
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

def get_git_commit_id():
    try:
        # Run the git command to get the current commit ID
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        # Decode from bytes to string and get the first 7 characters
        commit_id = commit_id.decode('utf-8')[:7]
        return commit_id
    except subprocess.CalledProcessError:
        # Handle errors if the git command fails
        print("An error occurred while trying to retrieve the Git commit ID.")
        return None
    
def get_current_date():
    # Get the current date
    current_date = datetime.now()
    # Format the date as a string (e.g., "YYYY-MM-DD")
    date_string = current_date.strftime("%Y-%m-%d_%H-%M")
    return date_string

def get_model_path():
    return get_current_date() + '_' + get_git_commit_id()
from collections.abc import Iterable
from tensorboardX import SummaryWriter
import os
import h5py

# ../summaries
# /Volumes/jwarren/notebooks_paper/summaries

class LearningLogger:
    def __init__(self, objs=(), root='./', path=''):
        self.pointers = []
        self.log = dict()
        self.root = root
        self.profile_path = os.path.join(self.root, 'summaries', path)
        if not os.path.exists(self.profile_path):
            print("Making directory: ", self.profile_path)
            os.makedirs(self.profile_path)
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

    def to_hdf5(self, file, arr, name, append_axis=1):
        def create_slice(append_axis, new_data_shape, dataset_shape):
            # Initialize slices as full slices for each dimension
            slices = [slice(None)] * len(dataset_shape)
            # Replace the slice in the specified axis with the new range
            if append_axis < len(dataset_shape):
                new_index = dataset_shape[append_axis]  # New index to start from
                new_range = new_data_shape[append_axis]  # Range of new data
                slices[append_axis] = slice(new_index, new_index + new_range)

            return tuple(slices)

        compression_level = 5 if arr.dtype == bool else 3
        max_shape = tuple([size if axis!= append_axis else None for axis, size in enumerate(arr.shape)])
        chunk_size = arr.shape  # Example chunk size
        if name not in file:
            file.create_dataset(name, data=arr, maxshape=max_shape, chunks=chunk_size, compression='gzip', compression_opts=compression_level)
        else:
            old_shape = file[name].shape
            file[name].resize((file[name].shape[append_axis] + arr.shape[append_axis]), axis=append_axis)
            append_slice = create_slice(1, arr.shape, old_shape)
            file[name][append_slice] = arr 


    def save_data(self, fname='data'):
        self.get_logs()
        with h5py.File(self.hdf5_path, 'a') as file:  # Open file in append mode
            self.to_hdf5(file, self.inputs_hist.astype(bool), 'inputs')
            self.to_hdf5(file, self.targets_hist.astype(bool), 'targets')
            self.to_hdf5(file, self.choices_hist, 'choices')
            self.to_hdf5(file, self.ground_truth_hist.astype(bool), 'ground_truth')
            self.to_hdf5(file, self.p_A_hist, 'p_A_high')
            self.to_hdf5(file, self.hidden_hist, 'hidden')

def is_iterable(obj):
    return isinstance(obj, Iterable)
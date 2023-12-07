import torch
import numpy as np
import pickle
import os
import h5py
from time import time
from torch.utils.tensorboard import SummaryWriter

class LearningLogger:
    def __init__(self, Conf):
        self.reset()
        self.save_dir = Conf.save_dir
        self.config = Conf
        self.writer = SummaryWriter('model/runs/experiment_name')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.hdf5_path = os.path.join(self.save_dir, 'data.h5')
        if os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)
        # self.hdf5_file = h5py.File(self.hdf5_path, 'w')  # Open in write mode

    def reset(self):
        self.a_accuracy_steps = 0
        self.x_accuracy_steps = 0
        self.n_trials = 0

        self.targets_hist_buffer = []
        self.inputs_hist_buffer = []
        self.choices_hist_buffer = []
        self.ground_truth_hist_buffer = []
        self.p_A_high_hist_buffer = []
        self.hidden_hist_buffer = []

    def log(self, logits=None, targets=None, inputs=None, ground_truth=None, p_A_high=None, hidden=None):
        self.n_trials += 1

        if logits is not None:
            # accuracy_all, accuracy_steps, accuracy_pair = self.compute_trial_accuracy(logits, targets, inputs)

            # self.a_accuracy_steps = ( self.a_accuracy_steps * (self.n_trials - 1) + accuracy_steps ) / self.n_trials
            # self.accuracy_all = ( self.accuracy_all * (self.n_trials - 1) + accuracy_all ) / self.n_trials
            # self.accuracy_pair = ( self.accuracy_pair * (self.n_trials - 1) + accuracy_pair ) / self.n_trials
            # self.choices_hist_buffer.append(torch.nn.functional.softmax(logits, dim=-1).numpy())  
            self.choices_hist_buffer.append(logits.numpy())        

        if inputs is not None:
            self.inputs_hist_buffer.append(inputs.numpy())
        if targets is not None:
            self.targets_hist_buffer.append(targets.numpy())
        if ground_truth is not None:
            self.ground_truth_hist_buffer.append(ground_truth)
        if p_A_high is not None:
            self.p_A_high_hist_buffer.append(p_A_high)
        if hidden is not None:
            self.hidden_hist_buffer.append(hidden)

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

        # self.choices.append()

    def get_data(self):
        self.inputs_hist = np.concatenate(self.inputs_hist_buffer, axis=1)
        self.targets_hist = np.concatenate(self.targets_hist_buffer, axis=1)
        self.ground_truth_hist = np.concatenate(self.ground_truth_hist_buffer, axis=1)
        self.p_A_hist = np.concatenate(self.p_A_high_hist_buffer, axis=1)
        self.hidden_hist = np.concatenate(self.hidden_hist_buffer, axis=1)
        
        if len(self.choices_hist_buffer):
            self.choices_hist = np.concatenate(self.choices_hist_buffer, axis=1)
            # choices_idx = 0 if not self.config.predict_x else self.config.state_dim
            a_correct = np.argmax(self.choices_hist[:,:,-self.config.action_dim:], axis=-1) == np.argmax(self.targets_hist[:,:,-self.config.action_dim:], axis=-1)
            # reshape into (batch_size, num_trial, num_trial_steps)
            a_correct = np.reshape(a_correct, (a_correct.shape[0], a_correct.shape[1] // self.config.trial_len, self.config.trial_len))
            self.a_accuracy_steps = 100 * np.sum(a_correct, axis=(0, 1)) / (a_correct.shape[0] * a_correct.shape[1])

            if self.config.predict_x:
                x_correct = np.argmax(self.choices_hist[:,:,:-self.config.action_dim], axis=-1) == np.argmax(self.targets_hist[:,:,:-self.config.action_dim], axis=-1)
                # reshape into (batch_size, num_trial, num_trial_steps)
                x_correct = np.reshape(x_correct, (x_correct.shape[0], x_correct.shape[1] // self.config.trial_len, self.config.trial_len))
                self.x_accuracy_steps = 100 * np.sum(x_correct, axis=(0, 1)) / (a_correct.shape[0] * a_correct.shape[1])
        # acc_all = 100 * np.sum(correct_all, axis=(0,1)) / (correct_all.shape[0] * correct_all.shape[1])
        # acc_steps = []
        # for i in range(self.inputs_hist_buffer[0].shape[1]):
        #     correct_step = correct[:, i::4]
        #     acc_step = 100 * np.sum(correct_step) / correct_step.size 
        #     acc_steps.append(acc_step)

        return

    def print(self):
        # trial_stages = ['rew', 'delay', 'init', 'init', 'choice']
        a_trial_stages = ['nothing' for _ in range(self.config.trial_len)]
        a_trial_stages[self.config.init_choice_step] = 'init'
        a_trial_stages[self.config.ab_choice_step] = 'choice'

        x_trial_stages = ['nothing' for _ in range(self.config.trial_len)]
        x_trial_stages[self.config.init_step] = 'init'
        x_trial_stages[self.config.r_step] = 'reward'
        x_trial_stages[self.config.a_step] = 'a'
        x_trial_stages[self.config.b_step] = 'b'
        print('Accuracy each step (a):\t' + ',\t'.join([f'{t}: {a:.1f}' for a, t in zip(self.a_accuracy_steps, a_trial_stages)]))
        if self.config.predict_x:
            print('Accuracy each step (x):\t' + ',\t'.join([f'{t}: {a:.1f}' for a, t in zip(self.x_accuracy_steps, x_trial_stages)]))

    def save_data(self, fname='data'):
        start = time()
        with h5py.File(self.hdf5_path, 'a') as file:  # Open file in append mode
            self.to_hdf5(file, self.inputs_hist.astype(bool), 'inputs')
            self.to_hdf5(file, self.targets_hist.astype(bool), 'targets')
            self.to_hdf5(file, self.choices_hist, 'choices')
            self.to_hdf5(file, self.ground_truth_hist.astype(bool), 'ground_truth')
            self.to_hdf5(file, self.p_A_hist, 'p_A_high')
            self.to_hdf5(file, self.hidden_hist, 'hidden')
        end = time()
        print(f'hdf5 save time: {end - start:.1f}')

        # if targets is not None:
        #     self.to_hdf5(targets.numpy(), 'targets')
        # if ground_truth is not None:
        #     self.to_hdf5(ground_truth, 'ground_truth')
        # if p_A_high is not None:
        #     self.to_hdf5(p_A_high, 'p_A_high')
        # if hidden is not None:
        #     self.to_hdf5(hidden.numpy(), 'hidden') 
    #     self.get_data()

        # fpath = os.path.join(self.save_dir, fname)
        # print(fpath)

        # data_dict = {'inputs_hist': self.inputs_hist,
        #              'targets_hist': self.targets_hist,
        #              'choices_hist': self.choices_hist,
        #              'ground_truth': self.ground_truth_hist,
        #              'p_A_hist': self.p_A_hist,
        #              'hidden': self.hidden_hist,
        #              'accuracy_steps': self.accuracy_steps}

        start = time()
        # np.savez(fpath, self.inputs_hist.astype(bool), self.targets_hist.astype(bool), self.choices_hist, self.ground_truth_hist.astype(bool), self.p_A_hist, self.hidden_hist, self.accuracy_steps)
        end = time()
        print(f'np save time: {end - start:.1f}')
        return

    def compute_trial_accuracy(self, logits, targets, inputs):
        # correct = torch.argmax(logits, axis=-1) == torch.argmax(targets, axis=-1)

        # acc_step = 100 * torch.sum(correct, axis=0, dtype=torch.float32) / correct.shape[0]

        # correct_all = torch.all(correct, axis=-1)
        # acc_all = 100 * torch.sum(correct_all, dtype=torch.float32) / correct.shape[0]

        # one_hot_choice = torch.nn.functional.one_hot(torch.argmax(logits[:, -1, :-1], axis=-1), num_classes=logits.size(-1))
        # two_hot_choices = inputs[:, -1, :-self.config.output_dim]
        # correct_pair = torch.sum(torch.logical_and(one_hot_choice, two_hot_choices), axis=-1)
        # acc_pair = 100 * torch.sum(correct_pair, axis=0, dtype=torch.float32) / correct.shape[0]

        # return acc_all, acc_step, acc_pair
        return np.nan, np.nan, np.nan

    def get_all_accuracies(self, logits, targets):
        splits = ['x', 'r', 'a']
        steps_split = [[self.config.init_step-1, self.config.a_step-1, self.config.b_step-1],
                       [self.config.r_step-1],
                       [self.config.init_choice_step, self.config.ab_choice_step]]
        
        dim_splits = [self.config.x_dim, self.config.r_dim, self.config.a_dim]
        logits_split = torch.split(logits, dim_splits, dim=-1)
        targets_split = torch.split(targets, dim_splits, dim=-1)

        accuracies = dict()
        for split, steps, logit, target in zip(splits, steps_split, logits_split, targets_split):
            accuracies[split] = self.get_accuracy_steps(logit, target, steps)

        return accuracies, steps_split
            
    def get_accuracy_steps(self, logits, targets, steps=None):
        if steps is None:
            steps = [i for i in range(self.config.trial_len)]

        T = logits.shape[self.config.t_ax]
        logits_steps = logits.reshape((self.config.batch_size, T//self.config.trial_len, self.config.trial_len, logits.shape[-1]))[:, :, steps, :]
        targets_steps = targets.reshape((self.config.batch_size, T//self.config.trial_len, self.config.trial_len, targets.shape[-1]))[:, :, steps, :]

        probabilities_steps = torch.nn.functional.softmax(logits_steps, dim=-1)
        correct = torch.argmax(probabilities_steps, dim=-1) == torch.argmax(targets_steps, dim=-1)

        return correct.sum(dim=(0,1)).float() / (correct.size(0) * correct.size(1))
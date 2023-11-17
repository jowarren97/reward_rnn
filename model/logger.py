import torch
import numpy as np
import pickle
import os


class LearningLogger:
    def __init__(self, Conf):
        self.reset()
        self.save_dir = Conf.save_dir
        self.config = Conf

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir = Conf.save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def reset(self):
        self.accuracy_steps = 0
        self.accuracy_all = 0
        self.accuracy_pair = 0
        self.n_trials = 0

        self.targets_hist_buffer = []
        self.inputs_hist_buffer = []
        self.choices_hist_buffer = []
        self.ground_truth_hist_buffer = []
        self.p_A_high_hist_buffer = []

    def log(self, logits=None, targets=None, inputs=None, ground_truth=None, p_A_high=None):
        self.n_trials += 1

        if logits is not None:
            accuracy_all, accuracy_steps, accuracy_pair = self.compute_trial_accuracy(logits, targets, inputs)

            self.accuracy_steps = ( self.accuracy_steps * (self.n_trials - 1) + accuracy_steps ) / self.n_trials
            self.accuracy_all = ( self.accuracy_all * (self.n_trials - 1) + accuracy_all ) / self.n_trials
            self.accuracy_pair = ( self.accuracy_pair * (self.n_trials - 1) + accuracy_pair ) / self.n_trials
            self.choices_hist_buffer.append(torch.nn.functional.softmax(logits, dim=-1).numpy())

        if inputs is not None:
            self.inputs_hist_buffer.append(inputs.numpy())
        if targets is not None:
            self.targets_hist_buffer.append(targets.numpy())
        if ground_truth is not None:
            self.ground_truth_hist_buffer.append(ground_truth)
        if p_A_high is not None:
            self.p_A_high_hist_buffer.append(p_A_high)

        # self.choices.append()

    def get_data(self):
        self.inputs_hist = np.concatenate(self.inputs_hist_buffer, axis=1)
        self.targets_hist = np.concatenate(self.targets_hist_buffer, axis=1)
        self.ground_truth_hist = np.concatenate(self.ground_truth_hist_buffer, axis=1)
        self.p_A_hist = np.concatenate(self.p_A_high_hist_buffer, axis=1)
        
        if len(self.choices_hist_buffer):
            self.choices_hist = np.concatenate(self.choices_hist_buffer, axis=1)
            correct = np.argmax(self.choices_hist, axis=-1) == np.argmax(self.targets_hist, axis=-1)

            # reshape into (batch_size, num_trial, num_trial_steps)
            correct = np.reshape(correct, (correct.shape[0], correct.shape[1] // 5, 5))
            correct_all = np.all(correct, axis=-1)

            self.accuracy_steps = 100 * np.sum(correct, axis=(0, 1)) / (correct.shape[0] * correct.shape[1])
        # acc_all = 100 * np.sum(correct_all, axis=(0,1)) / (correct_all.shape[0] * correct_all.shape[1])
        # acc_steps = []
        # for i in range(self.inputs_hist_buffer[0].shape[1]):
        #     correct_step = correct[:, i::4]
        #     acc_step = 100 * np.sum(correct_step) / correct_step.size 
        #     acc_steps.append(acc_step)

        return

    def print(self):
        trial_stages = ['rew', 'delay', 'init', 'init', 'choice']
        print('Accuracy all steps:\t' + f'{self.accuracy_all:.1f}')
        print(
            'Accuracy each step:\t' + ',\t'.join([f'{t}: {a:.1f}' for a, t in zip(self.accuracy_steps, trial_stages)]))
        print('Accuracy A or B:\t\t\t\t\t\t\t\t' + f'{self.accuracy_pair:.1f}')

    def save_data(self, fname='data'):
        self.get_data()

        fpath = os.path.join(self.save_dir, fname)
        print(fpath)

        data_dict = {'inputs_hist': self.inputs_hist,
                     'targets_hist': self.targets_hist,
                     'choices_hist': self.choices_hist,
                     'ground_truth': self.ground_truth_hist,
                     'p_A_hist': self.p_A_hist,
                     'accuracy_steps': self.accuracy_steps}

        np.savez(fpath, self.inputs_hist, self.targets_hist, self.choices_hist, self.ground_truth_hist, self.p_A_hist, self.accuracy_steps)


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

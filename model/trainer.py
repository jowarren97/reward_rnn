"""
#TODO:
- sort out grad clip params
- sort out x, r, a loss balance params
- sort out hyperparams method
"""

import torch
import numpy as np
import pickle
import os
import h5py
from time import time

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from environment import ReversalEnvironment
from itertools import permutations
from copy import copy
from random import shuffle

class Trainer():
    SAVE_OBJECT = "SAVE_OBJECT"
    SAVE_DICT = "SAVE_DICT"

    def __init__(self, model, config, logger=None):
        self.exit = False
        self.config = config
        self.model = model
        self.train_env, self.test_env = None, None
        # criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.logger = logger

    def get_envs(self):
        return self.get_train_env(), self.get_test_env()

    def get_train_env():
        raise NotImplementedError()
    
    def get_test_env():
        raise NotImplementedError()
    
    def train_for_single_epoch(model, logger=None):
        raise NotImplementedError()
    
    def save_model(self):
        if self.config.save_type == Trainer.SAVE_OBJECT:
            torch.save(self.model, self.model_path)
        elif self.config.save_type == Trainer.SAVE_DICT:
            torch.save(self.model.state_dict(), self.model_path)

    def save_model_log(self):
        raise NotImplementedError()
    
    def save_hyperparameters(self):
        raise NotImplementedError()
    
    def weight_loss(self, type='l2'):
        reg = torch.tensor(0.).to(self.config.dev)
        for param in self.model.parameters():
            reg += torch.norm(param, p=1 if type=='l1' else 2)
        return reg
    
    def activity_loss(self, hiddens, type='l2'):
        reg = torch.tensor(0.).to(self.config.dev)
        reg = torch.norm(hiddens, p=1 if type=='l1' else 2)
        return reg
    
    def output_loss(self, logits, targets):
        logits_ = torch.transpose(logits, 1, 2)
        targets_ = torch.transpose(targets, 1, 2)

        logits_x, logits_r, logits_a = self.train_env.split_data(logits_, dim=1)
        targets_x, targets_r, targets_a = self.train_env.split_data(targets_, dim=1)

        loss_x = self.criterion(logits_x, targets_x)
        loss_r = self.criterion(logits_r, targets_r)
        loss_a = self.criterion(logits_a, targets_a)

        return loss_x, loss_r, loss_a
    
    def loss(self, hiddens, logits, targets):
        loss_weight = self.config.weight_regularization * self.weight_loss()
        loss_activity = self.config.activity_regularization * self.activity_loss(hiddens)
        loss_x, loss_r, loss_a = self.output_loss(logits, targets)

        loss = loss_weight + loss_activity
        if self.config.predict_x:
            loss += loss_x + loss_a
        else:
            loss += loss_a

        return loss
    
    def train_for_single_epoch(self):
        epoch_loss = 0
        # Get the initial sequence for the epoch
        for batch_id in range(self.config.n_batches):
            hidden = None
            self.optimizer.zero_grad()

            with torch.no_grad():
                inputs, targets, _ = self.train_env.get_batch(self.config.num_trials, dropout=0.0)
                # Convert data to tensors
                data_tensor = inputs.to(dtype=self.config.dtype, device=self.config.dev)
                target_tensor = targets.to(dtype=self.config.dtype, device=self.config.dev)

            logits, hidden, hiddens = self.model(data_tensor, hidden)

            loss = self.loss(hiddens, logits, target_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss

    def on_training_start(self, save):
        if save:
            self.save_hyperparams()

    def on_training_complete(self, save):
        if save:
            self.save_model()
    
    def train(self, save=False):
        if save:
            os.makedirs(os.path.join(self.root, self.id))

        # self.on_training_start(save)

        for epoch in range(self.config.num_epochs):
            if self.exit:
                break
            # Train the model
            start_time = time()
            epoch_loss = self.train_for_single_epoch()
            end_time = time()
            epoch_duration = end_time - start_time
            self.logger.writer.add_scalar('Loss/train', epoch_loss, epoch)

            # logger.info(
            #     f"Completed epoch {epoch} with loss {epoch_loss} in {epoch_duration:.4f}s"
            # )
            # self.log["train_loss"].append(epoch_loss)
            # self.log["duration"].append(epoch_duration)

            self.on_epoch_complete(epoch, save=False)

        self.on_training_complete(save)

    def on_epoch_complete(self, epoch, save):
        if save:
            self.save_model_log()

        hidden = None
        self.logger.reset()
        with torch.no_grad():
            inputs, targets, groundtruths = self.test_env.get_batch(self.config.num_trials_test, dropout=0.0)
            # Convert data to tensors
            data_tensor = inputs.to(dtype=self.config.dtype, device=self.config.dev)
            target_tensor = targets.to(dtype=self.config.dtype, device=self.config.dev)
            logits, hidden, hiddens = self.model(data_tensor, hidden)

        # # store data in logger for later computation of accuracies
        if self.logger is not None:
            self.logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
                        groundtruths.cpu().detach(), self.train_env.optimal_agent.p_A_high.cpu().detach(), hiddens.cpu().detach())

        accuracies, steps = self.logger.get_all_accuracies(logits, targets)
        
        for i, (key, arr) in enumerate(accuracies.items()):
            for val, step in zip(arr, steps[i]):
                print(key, step, val.numpy())
                string = f'Accuracy/{key}_{step}'
                print(string)
                self.logger.writer.add_scalar(string, val.numpy(), epoch)

        loss = self.loss(hiddens, logits, target_tensor)
        print(f"Loss:\t\t\t{loss.item():.4f}")
        self.logger.get_data()
        self.logger.print()
            
        return



class ReversalTrainer(Trainer):
    def __init__(self, model, config, logger=None):
        super().__init__(model, config, logger)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_layouts, self.test_layouts, self.all_layouts = train_test_split()
        self.train_env, self.test_env = self.get_envs()

    def get_train_env(self):
        if self.train_env is None:
            self.train_env = ReversalEnvironment(self.config, self.train_layouts)
        return self.train_env
    
    def get_test_env(self):
        if self.train_env is None:
            self.train_env = ReversalEnvironment(self.config, self.test_layouts)
        return self.train_env




def generate_permutation_sets(port_dim):
    # Generate all permutations
    perms = permutations(range(port_dim), 3)

    # Dictionaries for initiation port sets and choice port sets
    # init_port_sets = {i: set() for i in range(port_dim)}
    # choice_port_sets = {frozenset([i, j]): set() for i in range(port_dim) for j in range(port_dim) if i != j}
    choice_port_sets = {frozenset([i, j]): [] for i in range(port_dim) for j in range(port_dim) if i != j}

    # Populate the sets
    for perm in perms:
        x, y, z = perm
        # init_port_sets[x].add(perm)
        # choice_port_sets[frozenset([y, z])].add(perm)
        choice_port_sets[frozenset([y, z])].append(perm)

    return choice_port_sets


def train_test_split(port_dim=9, train_ratio=0.8):
    new_dict = generate_permutation_sets(port_dim)
    reduced_dict = copy(new_dict)

    n_layouts = len(new_dict)
    train, test, all_perms, train_keys, test_keys = [], [], [], [], []
    n_overlap = 0
    
    keys_list = list(new_dict.keys())

    for key in keys_list:
        val = new_dict[key]
        # print(key, val)
        all_perms.append(val)
    shuffle(keys_list)

    while len(test) < n_layouts * (1 - train_ratio):
        for key in keys_list:
            val = new_dict[key]
            if sum([bool(key & k) for k in test_keys]) <= n_overlap:
                test_keys.append(key)
                test.append(list(val))
                del reduced_dict[key]
        keys_list = list(reduced_dict.keys())
        shuffle(keys_list)
        n_overlap += 1

    train_keys = [key for key in reduced_dict.keys()]
    train = [list(val) for val in reduced_dict.values()]
    train = [item for sublist in train for item in sublist]
    test = [item for sublist in test for item in sublist]
    all_perms = [item for sublist in all_perms for item in sublist]

    assert not any([bool(k_test == k_train) for k_test in test_keys for k_train in train_keys])

    all_perms = np.array([p for p in permutations(range(port_dim), 3)])

    return np.array(train), np.array(test), np.array(all_perms)


def accuracy(self, logits, targets):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    correct = torch.argmax(probabilities, dim=-1) == torch.argmax(targets, dim=-1)
    accuracy = correct_predictions.sum().float() / targets.size(0)
    
    return accuracy.item()

    

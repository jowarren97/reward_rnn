"""
#TODO:
- sort out grad clip params
- sort out x, r, a loss balance params
- sort out hyperparams method
"""

import torch
import numpy as np
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from environment import ReversalEnvironment
from itertools import permutations
from copy import copy
from random import shuffle
import logging
import sys
from logger import LearningLogger
from datetime import datetime
import subprocess

logger = logging.getLogger("trainer")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Trainer():
    SAVE_OBJECT = "SAVE_OBJECT"
    SAVE_DICT = "SAVE_DICT"

    def __init__(self, model, config, optimizer_func, scheduler_func=None, env_scheduler=None, optimizer_kwargs={}, scheduler_kwargs={}, root='../'):
        self.exit = False
        self.config = config
        self.model = model
        self.root = root
        self.path = get_path(self.config)
        print(self.path)
        self.model_path = os.path.join(self.root, 'run_data', self.path)
        self.train_env, self.test_env = None, None
        # criterion = nn.BCEWithLogitsLoss()
        # self.scheduler = Scheduler(start_lr=config.start_lr, end_lr=config.end_lr, decay_epochs=config.decay_epochs)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.start_lr)
        self.scheduler_func = scheduler_func
        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.env_scheduler = env_scheduler

        self.optimizer = optimizer_func(self.model.parameters(), **self.optimizer_kwargs)
        if self.scheduler_func is not None:
            self.scheduler = scheduler_func(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        self.log = {"train_loss": {"data": [], "tb": True, "save": False},
                    "test_loss" : {"data": [], "tb": True, "save": False},  
                    "epoch_time": {"data": [], "tb": True, "save": False},
                    "batch_time": {"data": [], "tb": True, "save": False},
                    "lr"        : {"data": [], "tb": True, "save": False},
                    "train_loss/x": {"data": [], "tb": True, "save": False},
                    "train_loss/r": {"data": [], "tb": True, "save": False},
                    "train_loss/a": {"data": [], "tb": True, "save": False},
                    "train_loss/weight": {"data": [], "tb": True, "save": False},
                    "train_loss/activity": {"data": [], "tb": True, "save": False}}
        
    def get_log(self):
        return self.log
    
    def reset_log(self):
        for key in self.log.keys():
            self.log[key]['data'] = []

    def get_envs(self):
        return self.get_train_env(), self.get_test_env()

    def get_train_env(self):
        raise NotImplementedError()
    
    def get_test_env(self):
        raise NotImplementedError()
    
    def save_model(self, epoch):
        if self.config.save_type == Trainer.SAVE_OBJECT:
            torch.save(self.model, os.path.join(self.model_path, str(epoch)))
        elif self.config.save_type == Trainer.SAVE_DICT:
            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'weights_' + str(epoch) + '.pth'))

    def save_env(self):
        raise NotImplementedError()

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
        logits_ = torch.transpose(logits, self.config.t_ax, -1)
        targets_ = torch.transpose(targets, self.config.t_ax, -1)

        logits_x, logits_r, logits_a = self.train_env.split_data(logits_, dim=self.config.t_ax)
        targets_x, targets_r, targets_a = self.train_env.split_data(targets_, dim=self.config.t_ax)

        x_t_mask = torch.tensor(self.config.x_t_mask).to(self.config.dev)
        logits_x = mask_tensor_by_timestep(logits_x, x_t_mask, t_ax=-1)
        targets_x = mask_tensor_by_timestep(targets_x, x_t_mask, t_ax=-1)

        a_t_mask = torch.tensor(self.config.a_t_mask).to(self.config.dev)
        logits_a = mask_tensor_by_timestep(logits_a, a_t_mask, t_ax=-1)
        targets_a = mask_tensor_by_timestep(targets_a, a_t_mask, t_ax=-1)

        r_t_mask = torch.tensor(self.config.r_t_mask).to(self.config.dev)
        logits_r = mask_tensor_by_timestep(logits_r, r_t_mask, t_ax=-1)
        targets_r = mask_tensor_by_timestep(targets_r, r_t_mask, t_ax=-1)

        loss_x = self.criterion(logits_x, targets_x)
        loss_r = self.criterion(logits_r, targets_r)
        loss_a = self.criterion(logits_a, targets_a)

        return loss_x, loss_r, loss_a
    
    def loss(self, hiddens, logits, targets):
        losses = {}
        t_mask = torch.tensor(self.config.a_t_mask).to(self.config.dev)

        loss_weight = self.config.weight_regularization * self.weight_loss()
        loss_activity = self.config.activity_regularization * self.activity_loss(hiddens)
        loss_x, loss_r, loss_a = self.output_loss(logits, targets)

        if self.config.predict_x:
            loss = loss_x + loss_a
        else:
            loss = loss_a

        loss += loss_activity

        if not isinstance(self.optimizer, torch.optim.AdamW):
            loss += loss_weight

        losses['weight'] = loss_weight.item()
        losses['activity'] = loss_activity.item()
        losses['x'] = loss_x.item()
        losses['r'] = loss_r.item()
        losses['a'] = loss_a.item()

        return loss, losses
    
    def train_for_single_epoch(self):
        epoch_loss = 0
        epoch_losses = {}
        # Get the initial sequence for the epoch
        for batch_id in range(self.config.n_batches):
            hidden = None
            self.optimizer.zero_grad()

            with torch.no_grad():
                inputs, targets, _ = self.train_env.get_batch(self.config.num_trials, self.env_scheduler.get_params())
                # Convert data to tensors
                data_tensor = inputs.to(dtype=self.config.dtype, device=self.config.dev)
                target_tensor = targets.to(dtype=self.config.dtype, device=self.config.dev)

            logits, hidden, hiddens = self.model(data_tensor, hidden)

            loss, losses = self.loss(hiddens, logits, target_tensor)

            loss.backward()
            # Check gradients
            # with torch.no_grad():
            #     for name, param in self.model.named_parameters():
            #         if param.requires_grad:
            #             print(f'{name} gradient: {torch.norm(param.grad, p=2)}')
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_losses = add_dictionaries(epoch_losses, losses)

        return epoch_loss, epoch_losses

    def on_training_start(self, save):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.save_env()
        if save:
            self.save_hyperparams()

    def on_training_complete(self, save):
        if save:
            self.save_model()
    
    def train(self, save=False):
        if save:
            os.makedirs(os.path.join(self.root, self.id))

        self.on_training_start(save)
        self.logger = LearningLogger([self, self.model, self.train_env, self.env_scheduler], root=self.root, path=self.path)

        for epoch in range(self.config.num_epochs):
            if self.exit:
                break
            # Train the model
            start_time = time()
            epoch_loss, epoch_losses = self.train_for_single_epoch()
            end_time = time()
            epoch_duration = end_time - start_time

            logger.info(
                f"Completed epoch {epoch} with loss {epoch_loss} in {epoch_duration:.4f}s"
            )
            self.log["train_loss"]["data"].append(epoch_loss)
            self.log["epoch_time"]["data"].append(epoch_duration)
            self.log["batch_time"]["data"].append(epoch_duration / self.config.n_batches)
            self.log["lr"]["data"].append(self.scheduler.get_last_lr())

            for key, val in epoch_losses.items():
                self.log["train_loss/" + key]["data"].append(val)

            save = epoch % self.config.save_model_interval == 0
            self.on_epoch_complete(epoch, save)

            if self.scheduler is not None: self.scheduler.step()
            if self.env_scheduler is not None: self.env_scheduler.step()

        self.on_training_complete(save)

    def on_epoch_complete(self, epoch, save):
        if save:
            self.save_model(epoch)

        print('Epoch', epoch+1)

        hidden = None
        with torch.no_grad():
            print('TRAIN')
            inputs, targets, groundtruths = self.train_env.get_batch(self.config.num_trials_test, self.env_scheduler.get_params())
            # Convert data to tensors
            data_tensor = inputs.to(dtype=self.config.dtype, device=self.config.dev)
            target_tensor = targets.to(dtype=self.config.dtype, device=self.config.dev)
            logits, hidden, hiddens = self.model(data_tensor, hidden)

        accuracies, steps = self.get_all_accuracies(logits, target_tensor)
        
        for i, (key, arr) in enumerate(accuracies.items()):
            for val, step in zip(arr, steps[i]):
                # print(key, step, val.numpy())
                string = f'trainAccuracy/{key}_{step}'
                # print(string)
                self.logger.writer.add_scalar(string, val, epoch)

        hidden = None
        with torch.no_grad():
            print('TEST')
            inputs, targets, groundtruths = self.test_env.get_batch(self.config.num_trials_test, self.env_scheduler.get_params())
            # Convert data to tensors
            data_tensor = inputs.to(dtype=self.config.dtype, device=self.config.dev)
            target_tensor = targets.to(dtype=self.config.dtype, device=self.config.dev)
            logits, hidden, hiddens = self.model(data_tensor, hidden)

        accuracies, steps = self.get_all_accuracies(logits, target_tensor)
        test_loss, test_losses = self.loss(hiddens, logits, target_tensor)
        self.log["test_loss"]["data"].append(test_loss)

        for i, (key, arr) in enumerate(accuracies.items()):
            for val, step in zip(arr, steps[i]):
                # print(key, step, val.numpy())
                string = f'testAccuracy/{key}_{step}'
                # print(string)
                self.logger.writer.add_scalar(string, val, epoch)

        self.logger.get_logs()
        self.logger.to_tensorboard(epoch)
        self.logger.reset_logs()
            
        return
    
    def get_all_accuracies(self, logits, targets):
        splits = ['x', 'r', 'a']
        # steps_split = [[self.config.init_step-1, self.config.a_step-1, self.config.b_step-1],
        #                 [self.config.r_step-1],
        #                 [self.config.init_choice_step, self.config.ab_choice_step]]
        steps_split = [[i for i in range(self.config.trial_len)] for j in range(3)]
        
        dim_splits = [self.config.x_dim, self.config.r_dim, self.config.a_dim]
        logits_split = torch.split(logits, dim_splits, dim=-1)
        targets_split = torch.split(targets, dim_splits, dim=-1)

        accuracies = dict()
        for split, steps, logit, target in zip(splits, steps_split, logits_split, targets_split):
            accuracies[split] = self.get_accuracy_steps(logit, target, steps)

        # for i, (key, arr) in enumerate(accuracies.items()):
        #     for val, step in zip(arr, steps_split[i]):
        #         print(key, step, val.numpy())
        #         string = f'Accuracy/{key}_{step}'
        #         print(string)

        a_trial_stages = ['nothing' for _ in range(self.config.trial_len)]
        if self.config.init_choice_step is not None: a_trial_stages[self.config.init_choice_step] = 'init'
        a_trial_stages[self.config.ab_choice_step] = 'choice'

        x_trial_stages = ['nothing' for _ in range(self.config.trial_len)]
        if self.config.init_step is not None: x_trial_stages[self.config.init_step] = 'init'
        x_trial_stages[self.config.r_step] = 'reward'
        x_trial_stages[self.config.a_step] = 'a'
        x_trial_stages[self.config.b_step] = 'b'
        print('Accuracy each step (a):\t' + ',\t'.join([f'{t}: {a:.1f}' for a, t in zip(accuracies['a'], a_trial_stages)]))
        print('Accuracy each step (x):\t' + ',\t'.join([f'{t}: {a:.1f}' for a, t in zip(accuracies['x'], x_trial_stages)]))

        return accuracies, steps_split
        
    def get_accuracy_steps(self, logits, targets, steps=None):
        if steps is None:
            steps = [i for i in range(self.config.trial_len)]

        T = logits.shape[self.config.t_ax]
        logits_steps = logits.reshape((self.config.batch_size, T//self.config.trial_len, self.config.trial_len, logits.shape[-1]))[:, :, steps, :]
        targets_steps = targets.reshape((self.config.batch_size, T//self.config.trial_len, self.config.trial_len, targets.shape[-1]))[:, :, steps, :]

        probabilities_steps = torch.nn.functional.softmax(logits_steps, dim=-1)
        correct = torch.argmax(probabilities_steps, dim=-1) == torch.argmax(targets_steps, dim=-1)

        return 100.0 * correct.sum(dim=(0,1)).float() / (correct.size(0) * correct.size(1))



class ReversalTrainer(Trainer):
    def __init__(self, model, config, optimizer_func, scheduler_func, env_scheduler, optimizer_kwargs={}, scheduler_kwargs={}):
        super().__init__(model, config, optimizer_func, scheduler_func, env_scheduler, optimizer_kwargs, scheduler_kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_layouts, self.test_layouts, self.all_layouts = train_test_split()
        self.train_env, self.test_env = self.get_envs()

    def get_train_env(self):
        if self.train_env is None:
            self.train_env = ReversalEnvironment(self.config, self.train_layouts)
        return self.train_env
    
    def get_test_env(self):
        if self.test_env is None:
            self.test_env = ReversalEnvironment(self.config, self.test_layouts)
        return self.test_env
    
    def save_env(self):
        fname = os.path.join(self.model_path, 'train_test_split.npz')
        np.savez(fname, train=self.train_layouts, test=self.test_layouts, all=self.all_layouts)


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


def get_path(config):
    date, time = get_current_date()
    return os.path.join(date, get_git_commit_id(), get_model_path(config), time)

def get_model_path(config):
    path = '_'.join([   'p', str(config.reward_prob), 
                        'lr', f'{config.start_lr:.0e}-{config.end_lr:.0e}',
                        'batchsize', str(config.batch_size),
                        'h', str(config.hidden_dim), 
                        'wreg', f'{config.weight_regularization:.0e}',
                        'hreg', f'{config.activity_regularization:.0e}',
                        'thresh', str(config.threshold),
                        'wgain', f'{config.init_hh_weight_gain:.0e}',
                        'lrdecay', f'{config.decay_epochs}',
                        'dropoutdecay', f'{config.dropout_decay_epochs}'])
    
    if config.amsgrad: path += '_amsgrad'

    return path

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
    date_string = current_date.strftime("%Y-%m-%d")
    time_string = current_date.strftime("%H:%M")
    return date_string, time_string



"""
def __init__(
        self,
        root,
        model,
        n_epochs,
        batch_size,
        lr,
        optimizer_func=torch.optim.Adam,
        scheduler_func=None,
        device="cuda",
        dtype=torch.float,
        grad_clip_value=None,
        save_type="SAVE_DICT",
        id=None,
        optimizer_kwargs={},
        scheduler_kwargs={},
        loader_kwargs={}):

        self.root = root
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_func = optimizer_func
        self.scheduler_func = scheduler_func
        self.device = device
        self.dtype = dtype
        self.grad_clip_value = grad_clip_value
        self.save_type = save_type
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        # Instantiate housekeeping variables
        self.id = str(uuid.uuid4().hex) if id is None else id

        # Initialise the model
        self.model = self.model.to(device)
        if dtype == torch.float:
            self.model = self.model.float()
        elif dtype == torch.half:
            self.model = self.model.half()
        self.date = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

        self.model_path = os.path.join(self.root, 'run_data', get_model_path())

        self.optimizer = self.optimizer_func(self.model.parameters(), lr=self.lr)

        # self.optimizer = self.optimizer_func(
        #     self.model.parameters(), self.lr, **optimizer_kwargs
        # )
        # if self.scheduler_func is not None:
        #     self.scheduler = scheduler_func(self.optimizer, **self.scheduler_kwargs)
        # else:
        #     self.scheduler = None

        # # Register grad clippings
        # if self.grad_clip_type == Trainer.GRAD_VALUE_CLIP_PRE:
        #     for p in self.model.parameters():
        #         p.register_hook(
        #             lambda grad: torch.clamp(grad, -grad_clip_value, grad_clip_value)
        #         )

        self.config = config
        self.train_env, self.test_env = None, None

        self.exit = False

        self.log = {"train_loss": {"data": [], "tb": True, "save": False},
                    "test_loss" : {"data": [], "tb": True, "save": False},  
                    "epoch_time": {"data": [], "tb": True, "save": False},
                    "batch_time": {"data": [], "tb": True, "save": False}}
                    """

def add_dictionaries(dict1, dict2):
    result = {}

    # Check if both dictionaries are empty
    if not dict1 and not dict2:
        return result

    # Check if dict1 is empty
    if not dict1:
        return dict2

    # Check if dict2 is empty
    if not dict2:
        return dict1

    # Add common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    for key in common_keys:
        result[key] = dict1[key] + dict2[key]

    # Add remaining keys from dict1
    for key in set(dict1.keys()) - common_keys:
        result[key] = dict1[key]

    # Add remaining keys from dict2
    for key in set(dict2.keys()) - common_keys:
        result[key] = dict2[key]

    return result

def mask_tensor_by_timestep(x, t_mask, t_ax=1):
    if t_mask.ndim != 1:
        raise ValueError('t_mask must be 1D')
    if t_ax < 0:
        t_ax = x.ndim + t_ax
        
    T = x.shape[t_ax]
    trial_len = len(t_mask)

    dims = [d for d in x.shape]
    dims[t_ax] = T//trial_len
    dims.insert(t_ax+1, trial_len)

    x_steps = x.reshape(dims)
    t_mask = t_mask.reshape([1 if dim != len(t_mask) else dim for dim in x_steps.shape])
    x_steps = x_steps * t_mask

    x_masked = x_steps.reshape(x.shape)

    return x_masked


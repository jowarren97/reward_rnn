import torch
from bayes_new import BayesAgent
from random import shuffle
from itertools import permutations
import numpy as np
from copy import copy

class Environment(object):
    """
    Base class for different environments.

    Attributes:
        config (object): Configuration object containing environment settings.
    """

    def __init__(self, config):
        """
        Initialize the environment with a given configuration.

        Args:
            config (object): Configuration object for the environment.
        """
        self.config = config
        self.dev = self.config.env_dev

        self.log = {'inputs'        : {"data": [], "tb": False, "save": True},
                    'targets'       : {"data": [], "tb": False, "save": True},
                    'groundtruths'  : {"data": [], "tb": False, "save": True}}
        
    def get_batch(self, num_trials=1):
        """
        Get a batch of data consisting of specified number of trials.

        Args:
            num_trials (int): Number of trials to retrieve.

        Returns:
            tuple: Tuple containing concatenated inputs, targets, and ground truths.
        """
        self.reset()

        inputs, targets, groundtruths = [], [], []
        t_ax = self.config.t_ax
        for i in range(num_trials):
            trial_input, trial_target, trial_groundtruth = self.get_trial()
            inputs.append(trial_input)
            targets.append(trial_target)
            groundtruths.append(trial_groundtruth)

        inputs_cat, targets_cat, groundtruths_cat = torch.cat(inputs, t_ax), torch.cat(targets, t_ax), torch.cat(groundtruths, t_ax)

        return self.check_all_conditions(inputs_cat, targets_cat, groundtruths_cat)

    def get_trial(self):
        """
        Get a single trial of data.

        Returns:
            tuple: Tuple containing inputs, targets, and ground truths for the trial.
        """
        return self.get_trial_inputs(), self.get_trial_targets(), self.get_trial_groundtruths()
    
    def get_trial_inputs(self):
        """Abstract method to get trial inputs."""
        raise NotImplementedError
    
    def get_trial_targets(self):
        """Abstract method to get trial targets."""
        raise NotImplementedError
    
    def get_trial_groundtruths(self):
        """Abstract method to get trial ground truths."""
        raise NotImplementedError
    
    def reset(self):
        """Abstract method to reset the environment."""
        raise NotImplementedError
    
    def reset_log(self):
        for key in self.log.keys():
            self.log[key]['data'] = []

    def get_log(self):
        return self.log

    def split_data(self, data):
        """
        Split the data into different components based on configured dimensions.

        Args:
            data (torch.Tensor): Data tensor to be split.

        Returns:
            tuple: Tuple of tensors split according to configured dimensions.

        Raises:
            ValueError: If the data cannot be split into the specified dimensions.
        """
        x_dim, r_dim, a_dim = self.config.x_dim, self.config.r_dim, self.config.a_dim
        try:
            return torch.split(data, [x_dim, r_dim, a_dim], axis=-1)
        except Exception as e:
            raise ValueError(f"Error splitting data: {e}")
        
    def check_all_conditions(self, inputs, targets, groundtruths):
        """
        Check all conditions for the data.

        Args:
            inputs (torch.Tensor): Input tensor to be checked.
            targets (torch.Tensor): Target tensor to be checked.
            groundtruths (torch.Tensor): Ground truth tensor to be checked.

        Returns:
            tuple: Tuple of tensors checked for all conditions.

        Raises:
            AssertionError: If any of the conditions are not met.
        """
        self.check_shapes(inputs, targets, groundtruths)
        self.check_types(inputs, targets, groundtruths)
        self.check_device(inputs, targets, groundtruths)
        # self.check_detached(inputs, targets, groundtruths)

        return inputs, targets, groundtruths
        
    def check_shapes(self, inputs, targets, groundtruths):
        t_ax, batch_ax = self.config.t_ax, self.config.batch_ax

        assert inputs.shape[batch_ax] == targets.shape[batch_ax] == groundtruths.shape[batch_ax], \
            f"Batch dimension of inputs, targets, and groundtruths must be the same. "
        assert inputs.shape[t_ax] == targets.shape[t_ax] == groundtruths.shape[t_ax], \
            f"Time dimension of inputs, targets, and groundtruths must be the same. "
        assert inputs.shape[-1] == self.config.input_dim, \
            f"Last dimension of inputs must be equal to input_dim. "
        assert targets.shape[-1] == self.config.output_dim, \
            f"Last dimension of targets must be equal to output_dim. "
        assert groundtruths.shape[-1] == self.config.output_dim, \
            f"Last dimension of groundtruths must be equal to output_dim. "
        return
    
    def check_types(self, inputs, targets, groundtruths):
        assert inputs.dtype == targets.dtype == groundtruths.dtype == self.config.dtype, \
            f"Data types of inputs, targets, and groundtruths must be the same as dtype. "
        return
    
    def check_device(self, inputs, targets, groundtruths):
        assert inputs.device == targets.device == groundtruths.device == self.dev, \
            f"Device of inputs, targets, and groundtruths must be the same as dev. "
        return
    
    def check_detached(self, inputs, targets, groundtruths):
        assert not inputs.requires_grad and not targets.requires_grad and not groundtruths.requires_grad, \
            f"Inputs, targets, and groundtruths must be detached. "
        return
    
class ReversalEnvironment(Environment):
    """
    Reversal environment class inheriting from Environment.

    Attributes:
        p (float): Probability attribute specific to the reversal environment.
    """

    def __init__(self, config, layouts):
        super().__init__(config)
        self.layouts = layouts
        self.init_vector, self.a_b_vector, self.a_vector, self.b_vector = self.generate_new_port_layout()

        self.last_ab_choice = torch.zeros((self.config.batch_size, self.config.a_dim), dtype=self.config.dtype, device=self.dev)
        self.last_reward = torch.randint(2, (self.config.batch_size, ), dtype=self.config.dtype, device=self.dev)

        self.optimal_agent = BayesAgent(config, self.a_vector, self.b_vector)
        # counters for reversals and block switches
        self.trials_since_reversal = torch.zeros(self.config.batch_size, dtype=torch.int32, device=self.dev)
        self.block_reversals = torch.zeros(self.config.batch_size, dtype=torch.int32, device=self.dev)
        # Fixed one-hot target based on two-hot input
        self.selected_two_hot_index = torch.randint(0, 2, (self.config.batch_size, 1), dtype=torch.int64, device=self.dev)  # which port is currently good (A=0, B=1)

        noise = torch.zeros((self.config.batch_size,), dtype=torch.int32, device=self.dev) if self.config.jitter == 0 \
            else torch.randint(-self.config.jitter, self.config.jitter + 1, (self.config.batch_size,), dtype=torch.int32, device=self.dev)
        # thresholds for reversals and block switches
        self.max_trials_since_reversal_jittered = self.config.max_trials_since_reversal + noise

        self.log = {**self.log, 
                    'p_A_high': {"data": [], "tb": False, "save": True}}

    def get_batch(self, num_trials, dropout=0.5):
        no_stim = torch.zeros((self.config.batch_size, self.config.x_dim), 
                                    dtype=self.config.dtype, device=self.dev)
        if self.config.no_stim_token: no_stim[:, -1] = 1
        
        no_reward = torch.zeros((self.config.batch_size, self.config.r_dim), 
                                dtype=self.config.dtype, device=self.dev)
        if self.config.no_reward_token: no_reward[:, -1] = 1
        
        no_action = torch.zeros((self.config.batch_size, self.config.a_dim), 
                                dtype=self.config.dtype, device=self.dev)
        if self.config.no_action_token: no_action[:, -1] = 1

        x_sequence, r_sequence, a_sequence, a_gt_sequence, p_A_sequence = [], [], [], [], []

        self.reset()

        for step in range(self.config.trial_len * num_trials):
            x, r, a, a_gt = no_stim, no_reward, no_action, no_action
            # stimulus, reward, action
            # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 2, 3, 4, 2, 4
            if (step - self.config.r_step) % self.config.trial_len == 0:
                r = torch.nn.functional.one_hot(torch.logical_not(self.last_reward).to(torch.int64), num_classes=self.config.r_dim)
                p_A = self.optimal_agent.update_beliefs(self.last_reward)

            if (step - self.config.init_step) % self.config.trial_len == 0:
                x = self.init_vector
            if (step - self.config.a_step) % self.config.trial_len == 0:
                x = self.a_vector
            if (step - self.config.b_step) % self.config.trial_len == 0:
                x = self.b_vector

            if (step - self.config.init_choice_step) % self.config.trial_len == 0:
                a = torch.nn.functional.pad(self.init_vector, (0, self.config.a_dim-self.init_vector.shape[-1]), mode='constant', value=0)
                a_gt = a
            if (step - self.config.ab_choice_step) % self.config.trial_len == 0:
                ab_choice = self.optimal_agent.choose_action()
                a = torch.nn.functional.pad(ab_choice, (0, self.config.a_dim-ab_choice.shape[-1]), mode='constant', value=0)
                a_gt = self.groundtruth_choice()
                self.compute_reward(a, a_gt)

            x_sequence.append(x)
            r_sequence.append(r)
            a_sequence.append(a)
            a_gt_sequence.append(a_gt)
            p_A_sequence.append(p_A)
            
            if (step - 1) % self.config.trial_len == 0:
                self.trials_since_reversal += 1
                self.check_and_switch_block()

        x, r, a, a_gt, p_A = stack_tensors([x_sequence, r_sequence, a_sequence, a_gt_sequence, p_A_sequence], axis=self.config.t_ax)
        x_shift, r_shift, a_shift , a_gt_shift, = shift_tensors([x, r, a, a_gt], axis=self.config.t_ax, shifts=[-1, -1, 1, 1])

        x_mask = [1 if i not in [self.config.init_step, self.config.a_step, self.config.b_step] else 1-dropout for i in range(self.config.trial_len)]
        x_masked, = mask_tensors([x], no_input_list=[no_stim], masks=[x_mask], axis=self.config.t_ax)
        
        inputs = torch.cat([x_masked, r, a_shift], dim=-1)
        targets = torch.cat([x_shift, r_shift, a], dim=-1)
        groundtruths = torch.cat([x_shift, r_shift, a_gt], dim=-1)
        # groundtruths = torch.cat([x_shift, r_shift, a], dim=-1)
        self.log['inputs']['data'].append(inputs)
        self.log['targets']['data'].append(targets)
        self.log['groundtruths']['data'].append(groundtruths)
        self.log['p_A_high']['data'].append(p_A)

        return inputs, targets, groundtruths

    def get_trial(self, groundtruths=None):
        """
        Get a single trial of data.

        Returns:
            tuple: Tuple containing inputs, targets, and ground truths for the trial.
        """
        # x_sequence = []
        # for step in range(self.config.trial_len):
        #     x_sequence.append()

        # return self.get_trial_inputs(), self.get_trial_targets(), self.get_trial_groundtruths()
        """Set the reward for the next input based on model's output, and get next input."""
        inputs, targets, groundtruths = self.get_trial_inputs(), self.get_trial_targets(), self.get_trial_groundtruths()
        self.post_trial_update(groundtruths)

        return inputs, targets, groundtruths
    
    def post_trial_update(self, groundtruths=None):
        if groundtruths is not None:
            action = self.optimal_agent.last_choice_onehot
            if self.config.predict_x:
                groundtruth_action = self.split_data(groundtruths)[2][:, self.config.ab_choice_step]
            else:
                groundtruth_action = groundtruths[:, self.config.ab_choice_step]
            trial_correct = torch.argmax(action, axis=-1) == torch.argmax(groundtruth_action, axis=-1)

            ps = torch.rand((self.config.batch_size,))
            reward_probabilistic = torch.where(trial_correct, ps < self.config.reward_prob, ps > self.config.reward_prob)
            # reward_probabilistic = reward
            self.optimal_agent.update_beliefs(reward_probabilistic)

            self.trials_since_reversal += 1
            self.check_and_switch_block()

            self.last_reward = reward_probabilistic
            self.last_ab_choice = self.optimal_agent.last_choice_onehot

    def groundtruth_choice(self):
        # Assuming self.a_b_vector is a PyTorch tensor
        two_hot_indices = (self.a_b_vector == 1).nonzero(as_tuple=True)[1].view(self.config.batch_size, 2)

        # Ensure that self.selected_two_hot_index is a PyTorch tensor
        # gathered = two_hot_indices.gather(1, self.selected_two_hot_index.unsqueeze(-1)).squeeze(-1)
        gathered = two_hot_indices.gather(1, self.selected_two_hot_index).squeeze(-1)

        # One-hot encoding in PyTorch
        ab_groundtruth = torch.nn.functional.one_hot(gathered, num_classes=self.config.a_dim)

        return ab_groundtruth

    def compute_reward(self, choice, true_choice):
        trial_correct = torch.argmax(choice, axis=-1) == torch.argmax(true_choice, axis=-1)

        ps = torch.rand((self.config.batch_size,), device=self.dev)
        reward_probabilistic = torch.where(trial_correct, ps < self.config.reward_prob, ps > self.config.reward_prob)

        self.last_reward = reward_probabilistic
    
    def check_and_switch_block(self):
        """Check elapsed trials since reversal and reverse if required. After, check total number of reversals and switch block if required"""
        # get batch mask for reversals
        max_trials_since_reversal_criterion = self.trials_since_reversal >= self.max_trials_since_reversal_jittered
        reversal_mask = max_trials_since_reversal_criterion

        # reversal if criteria met
        self.reverse(reversal_mask)
        # subsequently reset counters
        self.block_reversals[reversal_mask] += 1
        self.trials_since_reversal[reversal_mask] = 0

        if self.config.jitter == 0:
            noise = torch.zeros(self.config.batch_size, dtype=torch.int32, device=self.dev)
        else:
            noise = torch.randint(-self.config.jitter, self.config.jitter, (self.config.batch_size,), dtype=torch.int32, device=self.dev)
        # self.max_trials_since_reversal_jittered[reversal_mask] = self.config.max_trials_since_reversal + noise
        self.max_trials_since_reversal_jittered[:] = torch.where(reversal_mask, self.config.max_trials_since_reversal + noise,
                                                              self.max_trials_since_reversal_jittered)

        # get batch mask for block switches (i.e. changing of the 3 relevant ports)
        switch_mask = torch.logical_and(self.block_reversals % self.config.n_reversals == 0, self.block_reversals > 0)
        # switch if criteria met
        self.switch(switch_mask)
        # subsequently reset counters
        self.block_reversals[switch_mask] = 0
        # reset bayes agent probabilities
        self.optimal_agent.switch(switch_mask, self.a_vector, self.b_vector)

    def reverse(self, reversal_mask):
        """Reverse 'good' a or b port using a batch mask."""
        # Toggle target one-hot index
        if torch.any(reversal_mask):
            pass
        self.selected_two_hot_index[reversal_mask] = 1 - self.selected_two_hot_index[reversal_mask]

    def switch(self, switch_mask):
        """Switch blocks (i.e. the 3 active ports) using a batch mask."""
        init_vector_new, a_b_vector_new, a_vector_new, b_vector_new = self.generate_new_port_layout()

        switch_mask = switch_mask[:, np.newaxis]
        self.init_vector = torch.where(switch_mask, init_vector_new, self.init_vector)
        self.a_b_vector = torch.where(switch_mask, a_b_vector_new, self.a_b_vector)
        self.a_vector = torch.where(switch_mask, a_vector_new, self.a_vector)
        self.b_vector = torch.where(switch_mask, b_vector_new, self.b_vector)

    def reset(self):
        self.init_vector, self.a_b_vector, self.a_vector, self.b_vector = self.generate_new_port_layout()

        self.selected_two_hot_index = torch.randint(0, 2, (self.config.batch_size, 1), dtype=torch.int64, device=self.dev)  # which port is currently good (A=0, B=1)

        self.trials_since_reversal[:] = 0
        self.block_reversals[:] = 0

        self.last_ab_choice[:,:] = 0
        self.last_reward[:] = 0

        if self.config.jitter == 0:
            noise = torch.zeros((self.config.batch_size,), device=self.dev)
        else:
            noise = torch.randint(-self.config.jitter, self.config.jitter, (self.config.batch_size,), device=self.dev)

        self.max_trials_since_reversal_jittered[:] = self.config.max_trials_since_reversal + noise

        switch_mask = torch.ones((self.config.batch_size,), dtype=torch.bool, device=self.dev)
        self.optimal_agent.switch(switch_mask, self.a_vector, self.b_vector)

        self.reset_log()

    def generate_new_port_layout(self):
        """Generate random one-hot and two-hot vectors for batches."""
        # if self.eval:
        #     assert self.config.batch_size == len(self.all_layouts)
        #     perms = self.all_layouts
        # else:
        max_idx = len(self.layouts)
        idxs = torch.randint(0, max_idx, (self.config.batch_size,))
        perms = self.layouts[idxs]

        return self.perm_to_port_vectors(perms, self.config.x_dim)
    
    def split_data(self, data, dim=-1):
        """
        Split the data into different components based on configured dimensions.

        Args:
            data (torch.Tensor): Data tensor to be split.

        Returns:
            tuple: Tuple of tensors split according to configured dimensions.

        Raises:
            ValueError: If the data cannot be split into the specified dimensions.
        """
        x_dim, r_dim, a_dim = self.config.x_dim, self.config.r_dim, self.config.a_dim
        try:
            return torch.split(data, [x_dim, r_dim, a_dim], dim=dim)
        except Exception as e:
            raise ValueError(f"Error splitting data: {e}")


    def perm_to_port_vectors(self, perms, state_dim):
        perm_mat = np.array(perms)
        batch_size = perm_mat.shape[0]

        init_vector = torch.zeros((batch_size, state_dim), dtype=self.config.dtype, device=self.dev)
        init_vector[np.arange(batch_size), perm_mat[:, 0]] = 1  # initiation port

        a_vector = torch.zeros((batch_size, state_dim), dtype=self.config.dtype, device=self.dev)
        a_vector[np.arange(batch_size), perm_mat[:, 1]] = 1  # initiation port

        b_vector = torch.zeros((batch_size, state_dim), dtype=self.config.dtype, device=self.dev)
        b_vector[np.arange(batch_size), perm_mat[:, 2]] = 1  # initiation port

        a_b_vector = torch.zeros((batch_size, state_dim), dtype=self.config.dtype, device=self.dev)
        a_b_vector[np.arange(batch_size), perm_mat[:, 1]] = 1  # initiation port
        a_b_vector[np.arange(batch_size), perm_mat[:, 2]] = 1  # initiation port

        return init_vector, a_b_vector, a_vector, b_vector

    def port_vectors_to_perm(self, init_vector, a_vector, b_vector):
        batch_size = init_vector.shape[0]

        # Initialize the permutation matrix
        perm_mat = np.zeros((batch_size, 3), dtype=int)

        # Loop over each example in the batch
        for i in range(batch_size):
            # Find the indices where each vector is 1
            init_idx = np.where(init_vector[i] == 1)[0][0]
            a_idx = np.where(a_vector[i] == 1)[0][0]
            b_idx = np.where(b_vector[i] == 1)[0][0]

            # Construct the permutation
            perm_mat[i, 0] = init_idx
            perm_mat[i, 1] = a_idx
            perm_mat[i, 2] = b_idx

        return perm_mat


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


def check_train_test_split(curriculum, train):
    perms = port_vectors_to_perm(curriculum.current_block.init_vector, curriculum.current_block.a_vector, curriculum.current_block.b_vector)
    if train:
        assert not any([p in curriculum.current_block.test_layouts.tolist() for p in perms.tolist()])
        assert all([p in curriculum.current_block.train_layouts.tolist() for p in perms.tolist()])
    else:
        assert not any([p in curriculum.current_block.train_layouts.tolist() for p in perms.tolist()])
        assert all([p in curriculum.current_block.test_layouts.tolist() for p in perms.tolist()])

def stack_tensors(tensor_list, axis=0):
    return tuple([torch.stack(tensor, axis=axis) for tensor in tensor_list])

def shift_tensors(tensor_list, shifts=-1, axis=0):
    if type(shifts) is not list:
        shifts = [shifts] * len(tensor_list)
    return tuple([torch.roll(tensor, shift, dims=axis) for tensor, shift in zip(tensor_list, shifts)])

def mask_tensors(tensor_list, no_input_list, masks, axis=1, unmask_first=True):
    dev = tensor_list[0].device
    if type(masks) is not list:
        masks = [masks] * len(tensor_list)

    n_trials = tensor_list[0].shape[axis] // len(masks[0])
    n_batch = tensor_list[0].shape[0]
    T = tensor_list[0].shape[axis]

    masks = [torch.tensor(mask, device=dev) for mask in masks]
    initial_mask = torch.ones_like(masks[0])

    no_inputs = [torch.tile(null.unsqueeze(axis), (1, T, 1)) for null in no_input_list]
    assert all([null.shape == tensor.shape for null, tensor in zip(no_inputs, tensor_list)])

    if unmask_first:
        masks_extended = [torch.cat([torch.tile(initial_mask, (n_batch,1)), torch.tile(mask, (n_batch, n_trials-1))], dim=axis) for mask in masks]
    else:
        masks_extended = [torch.tile(mask, (n_batch, n_trials)) for mask in masks]

    masks_extended_bernoulli = [torch.bernoulli(mask_extended).to(torch.bool) for mask_extended in masks_extended]

    # return tuple([tensor * mask.view(1, -1, 1) for tensor, mask in zip(tensor_list, masks_extended_bernoulli)])
    return tuple([torch.where(mask.unsqueeze(-1), tensor, null) for tensor, mask, null in zip(tensor_list, masks_extended_bernoulli, no_inputs)])

    

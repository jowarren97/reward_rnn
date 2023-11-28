import numpy as np
from bayes import BayesAgent
from itertools import permutations
from copy import copy
from random import shuffle

epsilon = 1e-8


class Block:
    def __init__(self, config, starting_rewards=0, init_vector=None, a_b_vector=None, a_vector=None, b_vector=None, eval=False):
        self.batch_size = config.batch_size
        self.input_dim = config.input_dim  # n_ports + 1 (reward input)
        self.output_dim = config.output_dim
        self.state_dim = config.state_dim
        self.config = config
        self.train = True
        self.eval = eval
        self.train_layouts, self.test_layouts, self.all_layouts = train_test_split(port_dim=config.port_dim, train_ratio=0.8)

        self.reward_vector = np.zeros((self.batch_size, self.state_dim))
        self.reward_vector[:, -2] = starting_rewards
        self.reward_vector[:, -1] = 1 - starting_rewards

        if init_vector is None or a_b_vector is None:
            self.init_vector, self.a_b_vector, self.a_vector, self.b_vector = self.generate_new_port_layout()
        else:
            self.init_vector, self.a_b_vector, self.a_vector, self.b_vector = \
                init_vector, a_b_vector, a_vector, b_vector

        # Fixed one-hot target based on two-hot input
        self.selected_two_hot_index = np.random.choice([0, 1], size=(
            self.batch_size, 1))  # which port is currently good (A=0, B=1)

    def generate_new_port_layout(self):

        """Generate random one-hot and two-hot vectors for batches."""
        if self.eval:
            assert self.batch_size == len(self.all_layouts)
            perms = self.all_layouts
        else:
            max_idx = len(self.train_layouts) if self.train else len(self.test_layouts)
            idxs = np.random.choice(max_idx, size=self.batch_size)
            perms = self.train_layouts[idxs] if self.train else self.test_layouts[idxs]

        return perm_to_port_vectors(perms, self.state_dim)

        # all_indices = np.array([np.random.choice(self.config.port_dim, size=3, replace=False)
        #                         for _ in range(self.batch_size)])  # choose 3 random ports for task (from 9 possible)

        # init_vector = np.zeros((self.batch_size, self.state_dim))
        # init_vector[np.arange(self.batch_size), all_indices[:, 0]] = 1  # initiation port

        # a_b_vector = np.zeros((self.batch_size, self.state_dim))
        # a_b_vector[np.arange(self.batch_size), all_indices[:, 1]] = 1  # choice port
        # a_b_vector[np.arange(self.batch_size), all_indices[:, 2]] = 1  # choice port

        # a_vector = np.zeros((self.batch_size, self.state_dim))
        # b_vector = np.zeros((self.batch_size, self.state_dim))
        # a_vector[np.arange(self.batch_size), all_indices[:, 1]] = 1
        # b_vector[np.arange(self.batch_size), all_indices[:, 2]] = 1
        # return init_vector, a_b_vector, a_vector, b_vector
    
    def reset(self, train):
        self.train = train
        self.init_vector, self.a_b_vector, self.a_vector, self.b_vector = self.generate_new_port_layout()

    def reverse(self, reversal_mask):
        """Reverse 'good' a or b port using a batch mask."""
        # Toggle target one-hot index
        self.selected_two_hot_index[reversal_mask] = 1 - self.selected_two_hot_index[reversal_mask]

    def switch(self, switch_mask):
        """Switch blocks (i.e. the 3 active ports) using a batch mask."""
        init_vector_new, a_b_vector_new, a_vector_new, b_vector_new = self.generate_new_port_layout()

        switch_mask = switch_mask[:, np.newaxis]
        self.init_vector = np.where(switch_mask, init_vector_new, self.init_vector)
        self.a_b_vector = np.where(switch_mask, a_b_vector_new, self.a_b_vector)
        self.a_vector = np.where(switch_mask, a_vector_new, self.a_vector)
        self.b_vector = np.where(switch_mask, b_vector_new, self.b_vector)

    def get_data_sequence(self, action=None):
        """Generate an input sequence as per the block definition."""
        zero_x_vector = np.zeros((self.batch_size, self.state_dim))

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1

        init_action = self.init_vector[:, :self.output_dim]

        action = action if action is not None else do_nothing
        # sequence = np.stack([self.reward_vector, zero_vector, self.init_vector, self.a_b_vector], axis=1)
        x_sequence = self.get_x_sequence()

        a_sequence_list = [do_nothing for _ in range(self.config.trial_len)]
        a_sequence_list[0 if self.config.ab_choice_step==self.config.trial_len-1 else self.config.ab_choice_step+1] = action
        a_sequence_list[self.config.init_choice_step+1] = init_action
        a_sequence = np.stack(a_sequence_list, axis=1)

        # x_sequence_ = np.stack([self.reward_vector, zero_x_vector, self.init_vector, self.a_vector, self.b_vector], axis=1)
        # a_sequence_ = np.stack([action, do_nothing, do_nothing, init_action, do_nothing], axis=1)

        sequence = np.concatenate([x_sequence, a_sequence], axis=-1)

        return sequence

    # def get_target_sequence(self):
    #     """Generate target sequence containing: inaction, inaction, init port, good port"""
    #     target_sequence = []

    #     do_nothing = np.zeros((self.batch_size, self.output_dim))
    #     do_nothing[:, -1] = 1
    #     target_sequence.append(do_nothing)
    #     target_sequence.append(do_nothing)

    #     # initiation port choice
    #     target = self.init_vector[:, :-1]
    #     target_sequence.append(target)
    #     target_sequence.append(do_nothing)  # make this longer

    #     # Indices where values are 1 in two-hot matrix
    #     two_hot_indices = np.where(self.a_b_vector == 1)[1].reshape(self.batch_size, 2)
    #     # Use take_along_axis to gather elements from arr according to indices
    #     gathered = np.take_along_axis(two_hot_indices, self.selected_two_hot_index, axis=1)
    #     # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
    #     gathered = gathered.flatten()
    #     # Assume the maximum value in gathered is less than 10
    #     target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

    #     target_sequence.append(target)
    #     target_sequence = np.stack(target_sequence, axis=1)

    #     return target_sequence


class DataCurriculum:
    def __init__(self, config, eval=False):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.reward_prob = config.reward_prob
        self.batch_size = config.batch_size
        self.config = config

        self.current_block = Block(config, eval=eval)
        self.optimal_agent = BayesAgent(config, self.current_block.a_vector, self.current_block.b_vector)

        # counters for reversals and block switches
        self.trials_since_reversal = np.zeros(self.batch_size)
        self.block_reversals = np.zeros(self.batch_size)

        self.jitter = config.jitter
        self.max_trials_since_reversal = config.max_trials_since_reversal

        noise = np.zeros((self.batch_size,)) if self.jitter == 0 else np.random.randint(-self.jitter, self.jitter + 1,
                                                                                        (self.batch_size,))
        # thresholds for reversals and block switches
        self.max_trials_since_reversal_jittered = self.max_trials_since_reversal + noise
        self.n_reversals = config.n_reversals

    def reset(self, train):
        self.current_block.reset(train)

        self.trials_since_reversal[:] = 0
        self.block_reversals[:] = 0
        noise = np.random.randint(-self.jitter, self.jitter, self.batch_size,)
        self.max_trials_since_reversal_jittered[:] = self.max_trials_since_reversal + noise
        self.optimal_agent.switch(np.ones((self.batch_size,)), self.current_block.a_vector, self.current_block.b_vector)

    def get_data_sequence(self, action):
        """Get a data sequence based on the current block."""
        return self.current_block.get_data_sequence(action)

    def get_target_sequence(self):
        """Generate target sequence containing: inaction, inaction, init port, good port"""
        # target_sequence = []

        # do_nothing = np.zeros((self.batch_size, self.output_dim))
        # do_nothing[:, -1] = 1
        # target_sequence.append(do_nothing)
        # target_sequence.append(do_nothing)

        # # initiation port choice
        # target = self.current_block.init_vector[:, :self.output_dim]
        # target_sequence.append(target)
        # # target_sequence.append(target)  # make this longer
        # target_sequence.append(do_nothing)  # make this longer

        # # # Indices where values are 1 in two-hot matrix
        # # two_hot_indices = np.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # # # Use take_along_axis to gather elements from arr according to indices
        # # gathered = np.take_along_axis(two_hot_indices, self.optimal_agent.choose_action(), axis=1)
        # # # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        # # gathered = gathered.flatten()
        # # # Assume the maximum value in gathered is less than 10
        # # target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        # target = self.optimal_agent.choose_action()
        # target_sequence.append(target)

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1

        init_target = self.current_block.init_vector[:, :self.output_dim]
        
        choice_target = self.optimal_agent.choose_action()

        target_sequence_list = [do_nothing for _ in range(self.config.trial_len)]
        target_sequence_list[self.config.init_choice_step] = init_target
        target_sequence_list[self.config.ab_choice_step] = choice_target

        assert len(target_sequence_list) == self.config.trial_len
        
        target_sequence = np.stack(target_sequence_list, axis=1)

        return target_sequence


    def get_ground_truth_sequence(self):
        """Generate target sequence containing: inaction, inaction, init port, good port"""
        # target_sequence = []

        # do_nothing = np.zeros((self.batch_size, self.output_dim))
        # do_nothing[:, -1] = 1
        
        # target_sequence.append(do_nothing)
        # target_sequence.append(do_nothing)

        # # initiation port choice
        # target = self.current_block.init_vector[:, :self.output_dim]
        # target_sequence.append(target)
        # # target_sequence.append(target)  # make this longer
        # target_sequence.append(do_nothing)  # make this longer

        # # Indices where values are 1 in two-hot matrix
        # two_hot_indices = np.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # # Use take_along_axis to gather elements from arr according to indices
        # gathered = np.take_along_axis(two_hot_indices, self.current_block.selected_two_hot_index, axis=1)
        # # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        # gathered = gathered.flatten()
        # # Assume the maximum value in gathered is less than 10
        # target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        # target_sequence.append(target)

        # assert len(target_sequence) == self.config.trial_len
        
        # target_sequence_ = np.stack(target_sequence, axis=1)


        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1

        init_target = self.current_block.init_vector[:, :self.output_dim]
        
        # Indices where values are 1 in two-hot matrix
        two_hot_indices = np.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # Use take_along_axis to gather elements from arr according to indices
        gathered = np.take_along_axis(two_hot_indices, self.current_block.selected_two_hot_index, axis=1)
        # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        gathered = gathered.flatten()
        # Assume the maximum value in gathered is less than 10
        choice_target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        target_sequence_list = [do_nothing for _ in range(self.config.trial_len)]
        target_sequence_list[self.config.init_choice_step] = init_target
        target_sequence_list[self.config.ab_choice_step] = choice_target

        assert len(target_sequence_list) == self.config.trial_len
        
        target_sequence = np.stack(target_sequence_list, axis=1)

        return target_sequence


    def step(self, model_output=None, ground_truth=None):
        """Set the reward for the next input based on model's output, and get next input."""
        action = None
        if ground_truth is not None:
            if self.config.use_rnn_actions:
                action = model_output[:, self.config.ab_choice_step, :]
            else:
                action = self.optimal_agent.last_choice_onehot

            trial_correct = np.argmax(action, axis=-1) == np.argmax(ground_truth[:, self.config.ab_choice_step, :], axis=-1)

            ps = np.random.uniform(size=(self.batch_size,))
            reward_probabilistic = np.where(trial_correct, ps < self.reward_prob, ps > self.reward_prob)
            # reward_probabilistic = reward
            self.optimal_agent.update_beliefs(reward_probabilistic, choice=model_output[:, self.config.ab_choice_step, :] if self.config.use_rnn_actions else None)

            self.current_block.reward_vector[:, -2] = reward_probabilistic
            self.current_block.reward_vector[:, -1] = 1 - reward_probabilistic

            self.trials_since_reversal += 1

            self.check_and_switch_block()

        return self.get_data_sequence(action), self.get_target_sequence(), self.get_ground_truth_sequence()

    def check_and_switch_block(self):
        """Check elapsed trials since reversal and reverse if required. After, check total number of reversals and switch block if required"""
        # get batch mask for reversals
        max_trials_since_reversal_criterion = self.trials_since_reversal >= self.max_trials_since_reversal_jittered
        reversal_mask = max_trials_since_reversal_criterion

        if np.any(reversal_mask):
            pass
        # reversal if criteria met
        self.current_block.reverse(reversal_mask)
        # subsequently reset counters
        self.block_reversals[reversal_mask] += 1
        self.trials_since_reversal[reversal_mask] = 0
        noise = np.zeros((np.count_nonzero(reversal_mask),)) if self.jitter == 0 else np.random.randint(-self.jitter,
                                                                                                        self.jitter, (
                                                                                                            np.count_nonzero(
                                                                                                                reversal_mask),))
        self.max_trials_since_reversal_jittered[reversal_mask] = self.max_trials_since_reversal + noise

        # get batch mask for block switches (i.e. changing of the 3 relevant ports)
        switch_mask = np.logical_and(self.block_reversals % self.n_reversals == 0, self.block_reversals > 0)
        # switch if criteria met
        self.current_block.switch(switch_mask)
        # subsequently reset counters
        self.block_reversals[switch_mask] = 0
        # reset bayes agent probabilities
        self.optimal_agent.switch(switch_mask, self.current_block.a_vector, self.current_block.b_vector)


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


def perm_to_port_vectors(perms, state_dim):
    perm_mat = np.array(perms)
    batch_size = perm_mat.shape[0]

    init_vector = np.zeros((batch_size, state_dim))
    init_vector[np.arange(batch_size), perm_mat[:, 0]] = 1  # initiation port

    a_vector = np.zeros((batch_size, state_dim))
    a_vector[np.arange(batch_size), perm_mat[:, 1]] = 1  # initiation port

    b_vector = np.zeros((batch_size, state_dim))
    b_vector[np.arange(batch_size), perm_mat[:, 2]] = 1  # initiation port

    a_b_vector = np.zeros((batch_size, state_dim))
    a_b_vector[np.arange(batch_size), perm_mat[:, 1]] = 1  # initiation port
    a_b_vector[np.arange(batch_size), perm_mat[:, 2]] = 1  # initiation port

    if np.any(np.sum(a_b_vector, axis=-1)==1):
        pass

    return init_vector, a_b_vector, a_vector, b_vector


def port_vectors_to_perm(init_vector, a_vector, b_vector):
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


def check_train_test_split(curriculum, train):
    perms = port_vectors_to_perm(curriculum.current_block.init_vector, curriculum.current_block.a_vector, curriculum.current_block.b_vector)
    if train:
        assert not any([p in curriculum.current_block.test_layouts.tolist() for p in perms.tolist()])
        assert all([p in curriculum.current_block.train_layouts.tolist() for p in perms.tolist()])
    else:
        assert not any([p in curriculum.current_block.train_layouts.tolist() for p in perms.tolist()])
        assert all([p in curriculum.current_block.test_layouts.tolist() for p in perms.tolist()])


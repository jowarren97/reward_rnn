import numpy as np
from bayes import BayesAgent

epsilon = 1e-8

class Block:
    def __init__(self, config, starting_rewards=0, init_vector=None, a_b_vector=None):
        self.batch_size = config.batch_size
        self.input_dim = config.input_dim  # n_ports + 1 (reward input)
        self.output_dim = config.output_dim
        self.config = config

        self.reward_vector = np.zeros((self.batch_size, self.input_dim))
        self.reward_vector[:, -1] = starting_rewards

        if init_vector is None or a_b_vector is None:
            self.init_vector, self.a_b_vector = self.generate_new_port_layout()
        else:
            self.init_vector, self.a_b_vector = init_vector, a_b_vector

        # Fixed one-hot target based on two-hot input
        self.selected_two_hot_index = np.random.choice([0, 1], size=(self.batch_size, 1))  # which port is currently good (A=0, B=1)

    def generate_new_port_layout(self):
        """Generate random one-hot and two-hot vectors for batches."""
        all_indices = np.array([np.random.choice(self.input_dim - 1, size=3, replace=False)
                                for _ in range(self.batch_size)])  # choose 3 random ports for task (from 9 possible)

        init_vector = np.zeros((self.batch_size, self.input_dim))
        init_vector[np.arange(self.batch_size), all_indices[:, 0]] = 1  # initiation port

        a_b_vector = np.zeros((self.batch_size, self.input_dim))
        a_b_vector[np.arange(self.batch_size), all_indices[:, 1]] = 1  # choice port
        a_b_vector[np.arange(self.batch_size), all_indices[:, 2]] = 1  # choice port

        return init_vector, a_b_vector

    def reverse(self, reversal_mask):
        """Reverse 'good' a or b port using a batch mask."""
        # Toggle target one-hot index
        self.selected_two_hot_index[reversal_mask] = 1 - self.selected_two_hot_index[reversal_mask]

    def switch(self, switch_mask):
        """Switch blocks (i.e. the 3 active ports) using a batch mask."""
        init_vector_new, a_b_vector_new = self.generate_new_port_layout()

        switch_mask = switch_mask[:, np.newaxis]
        self.init_vector = np.where(switch_mask, init_vector_new, self.init_vector)
        self.a_b_vector = np.where(switch_mask, a_b_vector_new, self.a_b_vector)

    def get_data_sequence(self):
        """Generate an input sequence as per the block definition."""
        zero_vector = np.zeros((self.batch_size, self.input_dim))
        sequence = np.stack([self.reward_vector, zero_vector, self.init_vector, self.a_b_vector], axis=1)
        return sequence

    def get_target_sequence(self):
        """Generate target sequence containing: inaction, inaction, init port, good port"""
        target_sequence = []

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1
        target_sequence.append(do_nothing)
        target_sequence.append(do_nothing)

        # initiation port choice
        target = self.init_vector
        target_sequence.append(target)

        # Indices where values are 1 in two-hot matrix
        two_hot_indices = np.where(self.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # Use take_along_axis to gather elements from arr according to indices
        gathered = np.take_along_axis(two_hot_indices, self.selected_two_hot_index, axis=1)
        # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        gathered = gathered.flatten()
        # Assume the maximum value in gathered is less than 10
        target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        target_sequence.append(target)
        target_sequence = np.stack(target_sequence, axis=1)

        return target_sequence


class DataCurriculum:
    def __init__(self, config):
        self.input_dim = config.input_dim
        self.output_dim = config.input_dim
        self.reward_prob = config.reward_prob
        self.batch_size = config.batch_size
        self.config = config

        self.current_block = Block(config)
        self.optimal_agent = BayesAgent(config, self.current_block.a_b_vector)

        # counters for reversals and block switches
        self.trials_since_reversal = np.zeros(self.batch_size)
        self.block_reversals = np.zeros(self.batch_size)

        self.jitter = config.jitter
        self.max_trials_since_reversal = config.max_trials_since_reversal

        noise = np.zeros((self.batch_size,)) if self.jitter == 0 else np.random.randint(-self.jitter, self.jitter, (self.batch_size,))
        # thresholds for reversals and block switches
        self.max_trials_since_reversal_jittered = self.max_trials_since_reversal + noise
        self.n_reversals = config.n_reversals

    def get_data_sequence(self):
        """Get a data sequence based on the current block."""
        return self.current_block.get_data_sequence()

    def get_target_sequence(self):
        """Generate target sequence containing: inaction, inaction, init port, good port"""
        target_sequence = []

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1
        target_sequence.append(do_nothing)
        target_sequence.append(do_nothing)

        # initiation port choice
        target = self.current_block.init_vector
        target_sequence.append(target)

        # # Indices where values are 1 in two-hot matrix
        # two_hot_indices = np.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # # Use take_along_axis to gather elements from arr according to indices
        # gathered = np.take_along_axis(two_hot_indices, self.optimal_agent.choose_action(), axis=1)
        # # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        # gathered = gathered.flatten()
        # # Assume the maximum value in gathered is less than 10
        # target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        target = self.optimal_agent.choose_action()
        target_sequence.append(target)
        target_sequence = np.stack(target_sequence, axis=1)

        return target_sequence


    def get_ground_truth_sequence(self):
        """Generate target sequence containing: inaction, inaction, init port, good port"""
        target_sequence = []

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1
        target_sequence.append(do_nothing)
        target_sequence.append(do_nothing)

        # initiation port choice
        target = self.current_block.init_vector
        target_sequence.append(target)

        # Indices where values are 1 in two-hot matrix
        two_hot_indices = np.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # Use take_along_axis to gather elements from arr according to indices
        gathered = np.take_along_axis(two_hot_indices, self.current_block.selected_two_hot_index, axis=1)
        # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        gathered = gathered.flatten()
        # Assume the maximum value in gathered is less than 10
        target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        target_sequence.append(target)
        target_sequence = np.stack(target_sequence, axis=1)

        return target_sequence


    def step(self, model_output=None, ground_truth=None):
        """Set the reward for the next input based on model's output, and get next input."""
        if ground_truth is not None:
            if self.config.use_rnn_actions:
                trial_stages_correct = np.argmax(model_output[:, -2:, :], axis=-1) == np.argmax(ground_truth[:, -2:, :], axis=-1)
                reward = np.all(trial_stages_correct, axis=-1)
            else:
                reward = np.argmax(self.optimal_agent.last_choice_onehot, axis=-1) == np.argmax(ground_truth[:, -1, :], axis=-1)

            self.optimal_agent.update_beliefs(reward, choice=model_output[:, -1, :] if self.config.use_rnn_actions else None)

            self.current_block.reward_vector[:, -1] = reward
            self.trials_since_reversal += 1

            self.check_and_switch_block()

        return self.get_data_sequence(), self.get_target_sequence(), self.get_ground_truth_sequence()

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
        noise = np.zeros((np.count_nonzero(reversal_mask),)) if self.jitter == 0 else np.random.randint(-self.jitter, self.jitter, (np.count_nonzero(reversal_mask),))
        self.max_trials_since_reversal_jittered[reversal_mask] = self.max_trials_since_reversal + noise


        # get batch mask for block switches (i.e. changing of the 3 relevant ports)
        switch_mask = np.logical_and(self.block_reversals % self.n_reversals == 0, self.block_reversals > 0)
        # switch if criteria met
        self.current_block.switch(switch_mask)
        # subsequently reset counters
        self.block_reversals[switch_mask] = 0
        # reset bayes agent probabilities
        self.optimal_agent.switch(switch_mask, self.current_block.a_b_vector)

import numpy as np
from bayes import BayesAgent

epsilon = 1e-8

class Block:
    def __init__(self, batch_size=3, input_dim=10, starting_rewards=0, one_hot_vector=None, two_hot_vector=None):
        self.batch_size = batch_size
        self.input_dim = input_dim  # n_ports + 1 (reward input)
        self.reward_vector = np.zeros((self.batch_size, self.input_dim))
        self.reward_vector[:, -1] = starting_rewards
        self.trials_since_reversal = np.zeros((self.batch_size, 1))  # counter for trials since reversal
        self.block_reversals = np.zeros((self.batch_size,1))  # counter for number of reversals
        self.rewards_since_reversal = np.zeros((self.batch_size, 1))  # counter for rewards since reversal

        if one_hot_vector is None or two_hot_vector is None:
            self.one_hot_vector, self.two_hot_vector = self.generate_exclusive_vectors()
        else:
            self.one_hot_vector, self.two_hot_vector = one_hot_vector, two_hot_vector

        # Fixed one-hot target based on two-hot input
        self.selected_two_hot_index = np.random.choice([0, 1], size=(batch_size, 1))  # which port is currently good (A=0, B=1)

    def generate_exclusive_vectors(self):
        """Generate random one-hot and two-hot vectors for batches."""
        all_indices = np.array([np.random.choice(self.input_dim - 1, size=3, replace=False)
                                for _ in range(self.batch_size)])  # choose 3 random ports for task (from 9 possible)

        one_hot = np.zeros((self.batch_size, self.input_dim))
        one_hot[np.arange(self.batch_size), all_indices[:, 0]] = 1  # initiation port

        two_hot = np.zeros((self.batch_size, self.input_dim))
        two_hot[np.arange(self.batch_size), all_indices[:, 1]] = 1  # choice port
        two_hot[np.arange(self.batch_size), all_indices[:, 2]] = 1  # choice port

        return one_hot, two_hot

    def get_sequence(self):
        """Generate a sequence as per the block definition."""
        zero_vector = np.zeros((self.batch_size, self.input_dim))
        sequence = np.stack([self.reward_vector, zero_vector, self.one_hot_vector, self.two_hot_vector], axis=1)
        return sequence

    def reverse(self, reversal_mask):
        # Toggle target one-hot index
        self.selected_two_hot_index[reversal_mask] = 1 - self.selected_two_hot_index[reversal_mask]

    def switch(self, switch_mask):
        one_hot_vector_new, two_hot_vector_new = self.generate_exclusive_vectors()

        switch_mask = switch_mask[:, np.newaxis]
        self.one_hot_vector = np.where(switch_mask, one_hot_vector_new, self.one_hot_vector)
        self.two_hot_vector = np.where(switch_mask, two_hot_vector_new, self.two_hot_vector)

class DataCurriculum:
    def __init__(self, config):
        self.input_dim = config.input_dim
        self.output_dim = config.input_dim
        self.reward_prob = config.reward_prob
        self.batch_size = config.batch_size

        self.current_block = Block(batch_size=self.batch_size, input_dim=self.input_dim)
        self.optimal_agent = BayesAgent(batch_size=self.batch_size, p=self.reward_prob)

        self.rewards_since_reversal = np.zeros(self.batch_size)
        self.trials_since_reversal = np.zeros(self.batch_size)
        self.block_reversals = np.zeros(self.batch_size)

        self.jitter = config.jitter
        self.mean_trials_since_reversal = config.mean_trials_since_reversal

        noise = np.zeros((self.batch_size,)) if self.jitter == 0 else np.random.randint(-self.jitter, self.jitter, (self.batch_size,))
        self.max_trials_since_reversal_jittered = self.mean_trials_since_reversal + noise
        self.n_reversals = config.n_reversals
        self.reversal_threshold = 0.75

    def get_data_sequence(self):
        """Get a data sequence based on the current block."""
        return self.current_block.get_sequence()

    def get_target_sequence(self):
        """Generate the target sequence based on the current block's data."""
        # target_sequence = np.zeros((self.batch_size, 4, self.output_dim))
        target_sequence = []

        do_nothing = np.zeros((self.batch_size, self.output_dim))
        do_nothing[:, -1] = 1
        target_sequence.append(do_nothing)
        target_sequence.append(do_nothing)

        # initiation port choice
        target = self.current_block.one_hot_vector
        target_sequence.append(target)

        # Indices where values are 1 in two-hot matrix
        two_hot_indices = np.where(self.current_block.two_hot_vector == 1)[1].reshape(self.batch_size, 2)
        # Use take_along_axis to gather elements from arr according to indices
        gathered = np.take_along_axis(two_hot_indices, self.current_block.selected_two_hot_index, axis=1)
        # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        gathered = gathered.flatten()
        # Assume the maximum value in gathered is less than 10
        target = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        target_sequence.append(target)
        target_sequence = np.stack(target_sequence, axis=1)

        return target_sequence

    def set_and_get_next_input(self, model_output, target):
        """Set the reward for the next input based on model's output, and get next input."""

        trial_stages_correct = np.argmax(model_output[:, -2:, :], axis=-1) == np.argmax(target[:, -2:, :], axis=-1)

        reward = np.all(trial_stages_correct, axis=-1)

        self.current_block.reward_vector[:, -1] = reward
        self.rewards_since_reversal += reward
        self.trials_since_reversal += 1

        self.check_and_switch_block()

        return self.get_data_sequence()

    def check_and_switch_block(self):
        """Check if average reward crosses threshold and switch block if required"""
        # get batch mask for reversals
        max_trials_since_reversal_criterion = self.trials_since_reversal >= self.max_trials_since_reversal_jittered
        reversal_mask = max_trials_since_reversal_criterion
        # reversal if criteria met
        self.current_block.reverse(reversal_mask)
        # subsequently reset counters
        self.block_reversals[reversal_mask] += 1
        self.trials_since_reversal[reversal_mask] = 0
        self.rewards_since_reversal[reversal_mask] = 0
        noise = np.zeros((np.count_nonzero(reversal_mask),)) if self.jitter == 0 else np.random.randint(-self.jitter, self.jitter, (np.count_nonzero(reversal_mask),))
        self.max_trials_since_reversal_jittered[reversal_mask] = self.mean_trials_since_reversal + noise

        # get batch mask for block switches (i.e. changing of the 3 relevant ports)
        switch_mask = np.logical_and(self.block_reversals % self.n_reversals == 0, self.block_reversals > 0)
        # switch if criteria met
        self.current_block.switch(switch_mask)
        # subsequently reset counters
        self.block_reversals[switch_mask] = 0

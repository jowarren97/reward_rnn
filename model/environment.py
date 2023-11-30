import torch
from bayes import BayesAgent

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

    def get_batch(self, num_trials=1):
        """
        Get a batch of data consisting of specified number of trials.

        Args:
            num_trials (int): Number of trials to retrieve.

        Returns:
            tuple: Tuple containing concatenated inputs, targets, and ground truths.
        """
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
        return self.get_trial_inputs(), self.get_trial_targets(), self.get_trial_groundtruth()
    
    def get_trial_inputs(self):
        """Abstract method to get trial inputs."""
        raise NotImplementedError
    
    def get_trial_targets(self):
        """Abstract method to get trial targets."""
        raise NotImplementedError
    
    def get_trial_groundtruth(self):
        """Abstract method to get trial ground truth."""
        raise NotImplementedError
    
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
        # self.check_device(inputs, targets, groundtruths)
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
        assert inputs.dtype == targets.dtype == groundtruths.dtype == self.config.env.dtype, \
            f"Data types of inputs, targets, and groundtruths must be the same as dtype. "
        return
    
    def check_device(self, inputs, targets, groundtruths):
        assert inputs.device == targets.device == groundtruths.device == self.config.dev, \
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

    def __init__(self, config):
        super().__init__(config)
        self.a_vector, self.b_vector, self.init_vector = None, None, None
        self.optimal_agent = BayesAgent(config, self.a_vector, self.b_vector)
        # counters for reversals and block switches
        self.trials_since_reversal = torch.zeros(self.config.batch_size)
        self.block_reversals = torch.zeros(self.config.batch_size)

        noise = torch.zeros((self.config.batch_size,)) if self.jitter == 0 \
            else torch.random.randint(-self.config.jitter, self.config.jitter + 1, (self.config.batch_size,))
        # thresholds for reversals and block switches
        self.max_trials_since_reversal_jittered = self.config.max_trials_since_reversal + noise

    def get_trial(self, reward, last_action):
        """
        Get a single trial of data.

        Returns:
            tuple: Tuple containing inputs, targets, and ground truths for the trial.
        """

        inputs = self.get_trial_inputs(reward, last_action)

        self.optimal_agent.update_beliefs(reward)
        next_bayes_action = self.optimal_agent.choose_action()
        targets = self.get_trial_targets(reward, next_bayes_action)

        two_hot_indices = torch.where(self.current_block.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # Use take_along_axis to gather elements from arr according to indices
        gathered = torch.take_along_axis(two_hot_indices, self.current_block.selected_two_hot_index, axis=1)
        # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        gathered = gathered.flatten()
        # Assume the maximum value in gathered is less than 10
        ideal_action = torch.eye(self.config.action_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)

        groundtruths = self.get_trial_groundtruth(reward, ideal_action)

        groundtruth_choice = self.split_data(groundtruths)[2][:, self.config.ab_choice_step, :]
        trial_correct = torch.argmax(next_bayes_action, axis=-1) == torch.argmax(groundtruth_choice, axis=-1)
        ps = torch.random.uniform(size=(self.batch_size,))

        next_reward = torch.where(trial_correct, ps < self.reward_prob, ps > self.reward_prob)

        return inputs, targets, groundtruths, next_reward, next_bayes_action

    def get_trial_inputs(self, action, reward):
        """
        Get trial inputs specific to the reversal environment.

        Returns:
            torch.Tensor: Concatenated tensor of x, r, and a sequences.
        """
        x_sequence = self.get_x_sequence()
        r_sequence = self.get_r_sequence(reward)
        # apply a shift of the actions rightwards by 1
        a_sequence = self.get_a_sequence(action)
        a_sequence = torch.roll(a_sequence, shifts=1, dims=self.config.t_ax)

        return torch.cat([x_sequence, r_sequence, a_sequence], -1)
    
    def get_trial_targets(self, action, reward):
        """
        Get trial targets specific to the reversal environment.

        Returns:
            torch.Tensor: Concatenated tensor of x, r, and a sequences.
        """
        x_sequence = self.get_x_sequence()
        r_sequence = self.get_r_sequence(reward)
        a_sequence = self.get_a_sequence(action)

        return torch.cat([x_sequence, r_sequence, a_sequence], -1)
    
    def get_trial_groundtruth(self, action, reward):
        x_sequence = self.get_x_sequence()
        r_sequence = self.get_r_sequence(reward)
        a_sequence = self.get_a_sequence(action)

        return torch.cat([x_sequence, r_sequence, a_sequence], -1)
    
    def get_x_sequence(self):
        """Abstract method to get x sequence."""
        zero_x_vector = torch.zeros((self.config.batch_size, self.config.x_dim), 
                                    dtype=self.config.env.dtype, device=self.config.env.device)
        
        x_sequence_list = [zero_x_vector for _ in range(self.config.trial_len)]
        x_sequence_list[self.config.init_step] = self.init_vector
        x_sequence_list[self.config.a_step] = self.a_vector
        x_sequence_list[self.config.b_step] = self.b_vector

        return torch.stack(x_sequence_list, axis=self.config.t_ax)
       
    def get_r_sequence(self, reward):
        """Abstract method to get r sequence."""
        zero_r_vector = torch.zeros((self.config.batch_size, self.config.r_dim), 
                                     dtype=self.config.env.dtype, device=self.config.env.device)

        r_sequence_list = [zero_r_vector for _ in range(self.config.trial_len)]
        r_sequence_list[self.config.r_step] = reward

        return torch.stack(r_sequence_list, axis=self.config.t_ax)
    
    def get_a_sequence(self, ab_choice):
        """Abstract method to get a sequence."""
        no_action = torch.zeros((self.config.batch_size, self.config.a_dim), 
                                     dtype=self.config.env.dtype, device=self.config.env.device)
        no_action[:, -1] = 1

        init_action = self.init_vector

        a_sequence_list = [no_action for _ in range(self.config.trial_len)]

        a_sequence_list[self.config.ab_choice_step] = ab_choice
        a_sequence_list[self.config.init_choice_step] = init_action

        return torch.stack(a_sequence_list, axis=self.config.t_ax)
    
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

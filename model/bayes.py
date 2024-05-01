import torch

class BayesAgent:
    def __init__(self, config, a_vector, b_vector, p_init=0.5):
        self.config = config
        self.batch_size = config.batch_size
        self.p = config.reward_prob  # reward contingency for high port
        self.p_A_high = p_init * torch.ones((self.batch_size, 1), device=self.config.env_dev)  # batched initial beliefs
        self.last_choice = torch.randint(2, (self.batch_size, 1), device=self.config.env_dev)
        self.a_vector = a_vector
        self.b_vector = b_vector
        self.output_dim = config.action_dim
        self.p_init = p_init

    def update_beliefs(self, reward):
        """
        Update beliefs based on the choice made and the reward received.
        
        Args:
        - choice (tensor): Either 0 (A) or 1 (B). Size: batch_size
        - reward (tensor): Either 0 or 1. Size: batch_size
        """
        # choice_binary = np.where(self.last_choice if choice is None else choice, axis=-1)
        choice_binary = self.last_choice

        # Determine the likelihood based on the choice and reward
        p_data_given_A_high = torch.where(torch.logical_xor(choice_binary, torch.unsqueeze(reward, axis=1)), self.p, 1-self.p)
        p_data_given_B_high = 1 - p_data_given_A_high
        
        # Update the posterior beliefs
        p_A_high_post_trans = (1 - self.config.p_switch) * self.p_A_high + self.config.p_switch * (1 - self.p_A_high)

        # Compute the total probability of the evidence
        p_data = p_data_given_A_high * p_A_high_post_trans + p_data_given_B_high * (1 - p_A_high_post_trans)

        self.p_A_high = p_data_given_A_high * p_A_high_post_trans / p_data

        # # self.p_A_high = np.where(p_data>0, (p_data_given_A_high * self.p_A_high / p_data), 1-self.p_A_high)
        # new_belief = np.where(p_data>0, (p_data_given_A_high * self.p_A_high / p_data), 1-self.p_A_high)
        # self.p_A_high = self.config.alpha * new_belief + (1 - self.config.alpha) * ((self.p_A_high + 0.5) / 2)

        return self.p_A_high


    def switch(self, switch_mask, a_vector, b_vector):
        switch_mask = torch.unsqueeze(switch_mask, dim=-1)
        self.a_vector = torch.where(switch_mask, a_vector, self.a_vector)
        self.b_vector = torch.where(switch_mask, b_vector, self.b_vector)
        self.p_A_high = torch.where(switch_mask, self.p_init, self.p_A_high)

    def choose_action(self):
        """
        Choose an action based on the current beliefs.
        
        Returns:
        - tensor: Either 0 (A) or 1 (B). Size: batch_size
        """
        self.last_choice = torch.where(self.p_A_high > 0.5, 0, 1)

        self.last_choice_onehot = torch.where(self.p_A_high > 0.5, self.a_vector, self.b_vector)[:, :self.output_dim]
        # # Indices where values are 1 in two-hot matrix
        # two_hot_indices = np.where(self.a_b_vector == 1)[1].reshape(self.batch_size, 2)
        # # Use take_along_axis to gather elements from arr according to indices
        # gathered = np.take_along_axis(two_hot_indices, self.last_choice, axis=1)
        # # Since gathered will have shape (64, 1), you might want to flatten it to get a 1D array
        # gathered = gathered.flatten()
        # # Assume the maximum value in gathered is less than 10
        # self.last_choice_onehot = np.eye(self.output_dim)[gathered.astype(int)]  # This creates a one-hot encoded array of shape (64, 10)
 
        return self.last_choice_onehot  

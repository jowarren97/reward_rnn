import torch

class BayesAgent:
    def __init__(self, batch_size=3, p=0.8):
        self.batch_size = batch_size
        self.p = p  # reward contingency for high port
        self.p_A_high = torch.tensor([0.5] * batch_size)  # batched initial beliefs
    
    def update_beliefs(self, choice, reward):
        """
        Update beliefs based on the choice made and the reward received.
        
        Args:
        - choice (tensor): Either 0 (A) or 1 (B). Size: batch_size
        - reward (tensor): Either 0 or 1. Size: batch_size
        """

        # Determine the likelihood based on the choice and reward
        p_data_given_A_high = torch.where(torch.logical_xor(choice, reward), self.p, 1-self.p)
        p_data_given_B_high = 1 - p_data_given_A_high
        
        # Compute the total probability of the evidence
        p_data = p_data_given_A_high * self.p_A_high + p_data_given_B_high * (1 - self.p_A_high)
        
        # Update the posterior beliefs
        self.p_A_high = p_data_given_A_high * self.p_A_high / p_data

    def choose_action(self):
        """
        Choose an action based on the current beliefs.
        
        Returns:
        - tensor: Either 0 (A) or 1 (B). Size: batch_size
        """
        return torch.where(self.p_A_high > 0.5, 0, 1)   

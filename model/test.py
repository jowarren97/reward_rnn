from curriculum import DataCurriculum, check_train_test_split
from model import SimpleRNN
import torch.nn as nn
import torch
from config import Conf
from logger import LearningLogger
from time import time
import os

print("Using device: ", Conf.dev)

Conf.batch_size = 504
Conf.reward_prob = 0.8
Conf.hidden_dim = 256
load_dir = 'run_data_08_256_l2_1e6_l2_3e4'
# load_dir = 'run_data'
load_dir = os.path.join(os.getcwd(), load_dir)
print(load_dir)
Conf.save_dir = load_dir

# Initialize model, curriculum, loss function and optimizer
model = SimpleRNN(Conf)
model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_50000.pth')))

data_curriculum = DataCurriculum(Conf, eval=True)
logger = LearningLogger(Conf)

def step(model, num_trials, logger=None):
    """Do one epoch."""
    hidden = None
    # Get the initial sequence for the epoch
    next_input, next_target, ground_truth = data_curriculum.step()

    # Convert data to tensors
    data_tensor = torch.tensor(next_input, dtype=Conf.dtype, device=Conf.dev)
    target_tensor = torch.tensor(next_target, dtype=Conf.dtype, device=Conf.dev)
    t = 0
    s_ = time()
    for trial in range(num_trials):
        """Loop through trials (chunks of 5 timesteps)."""
        # Forward pass
        logits, hidden, hiddens = model(data_tensor, hidden)

        if Conf.sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            choices = torch.nn.functional.one_hot(dist.sample(), num_classes=Conf.input_dim)
        else:
            choices = logits

        # store data in logger for later computation of accuracies
        if logger is not None:
            logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
                       ground_truth, data_curriculum.optimal_agent.p_A_high, hiddens.cpu().detach())

        # Prepare next input based on the output and target (computes if reward should be recieved, 
        # and also whether reversal or block switch occurs)
        next_input, next_target, ground_truth = data_curriculum.step(choices.cpu().detach().numpy(), 
                                                          ground_truth)

        data_tensor = torch.tensor(next_input, dtype=Conf.dtype, device=Conf.dev)
        target_tensor = torch.tensor(next_target, dtype=Conf.dtype, device=Conf.dev)

    return


model.eval()

for i in range(3):
    data_curriculum.reset(train=False)
    logger.reset()
    step(model, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname='data_all')
    logger.print()

# logger.save_data(fname='data_all')

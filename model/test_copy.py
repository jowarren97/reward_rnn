from curriculum import train_test_split, check_train_test_split
from environment import ReversalEnvironment
from model import SimpleRNN
import torch
from config import Conf
from logger import LearningLogger
from time import time
import os
from tqdm import tqdm

print("Using device: ", Conf.dev)

Conf.batch_size = 504
Conf.hidden_dim = 256
Conf.num_epochs_test = 1
# load_dir = 'run_data_08_256_l2_1e6_l2_3e4'
load_dir = 'run_data_new'
# load_dir = 'run_data'
load_dir = os.path.join(os.getcwd(), load_dir)
print(load_dir)
Conf.save_dir = os.path.join(os.getcwd(), 'run_data_new_env')

# Initialize model, curriculum, loss function and optimizer
model = SimpleRNN(Conf)
model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_3000.pth')))

logger = LearningLogger(Conf)

train_layouts, test_layouts, all_layouts = train_test_split()
train_env, test_env, all_env = ReversalEnvironment(Conf, train_layouts), ReversalEnvironment(Conf, test_layouts), ReversalEnvironment(Conf, all_layouts)


def step(model, env, num_trials, logger=None):
    """Do one epoch."""
    hidden = None

    hidden = None
    # Get the initial sequence for the epoch
    with torch.no_grad():
        inputs, targets, groundtruths = env.get_batch(num_trials, dropout=0.0)
        # Convert data to tensors
        data_tensor = inputs.to(dtype=Conf.dtype, device=Conf.dev)
        target_tensor = targets.to(dtype=Conf.dtype, device=Conf.dev)

    """Loop through trials (chunks of 4 timesteps)."""
    # Forward pass
    logits, hidden, hiddens = model(data_tensor, hidden)

    # store data in logger for later computation of accuracies
    if logger is not None:
        logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
                    groundtruths, env.optimal_agent.p_A_high, hiddens.cpu().detach())

    return


model.eval()

for i in range(Conf.num_epochs_test):
    logger.reset()
    step(model, train_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.print()

    logger.reset()
    step(model, test_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.print()

    logger.reset()
    step(model, all_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname=f'data_all_{i}')
    logger.print()

# logger.save_data(fname='data_all')

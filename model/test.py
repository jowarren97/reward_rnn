from curriculum import DataCurriculum, check_train_test_split
from model import SimpleRNN
import torch.nn as nn
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
model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_5000.pth')))

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
    for trial in tqdm(range(num_trials)):
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

for i in range(Conf.num_epochs_test):
    data_curriculum = DataCurriculum(Conf, eval=False)
    data_curriculum.reset(train=True)     
    check_train_test_split(data_curriculum, train=True)
    logger.reset()
    step(model, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.print()

    data_curriculum = DataCurriculum(Conf, eval=False)
    data_curriculum.reset(train=False)     
    check_train_test_split(data_curriculum, train=False)
    logger.reset()
    step(model, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.print()

    data_curriculum = DataCurriculum(Conf, eval=True)
    data_curriculum.reset(train=False)
    logger.reset()
    step(model, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname=f'data_all_{i}')
    logger.print()

# logger.save_data(fname='data_all')

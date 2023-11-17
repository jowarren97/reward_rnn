"""
- function to move between A/B binary and one-hot representations
- mechanism to remove block switching (time with epoch starts only)
- general formulation of rnn: one for standard, one for groupRNN
- better logger
- curriculum class, maybe do in tensors not numpy?
- in curriculum, make action, state, reward formulation more clean and explicit
- held out layouts
"""

from curriculum import DataCurriculum, check_train_test_split
from model import SimpleRNN
import torch.nn as nn
import torch
from config import Conf
from logger import LearningLogger
from time import time

print("Using device: ", Conf.dev)

debug = False
if debug:
    Conf.batch_size = 1

# Initialize model, curriculum, loss function and optimizer
model = SimpleRNN(Conf)
data_curriculum = DataCurriculum(Conf)
logger = LearningLogger(Conf)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Conf.lr)
epoch_time = 0


def step(model, num_trials, logger=None):
    """Do one epoch."""
    hidden = None
    loss = 0
    # Get the initial sequence for the epoch
    next_input, next_target, ground_truth = data_curriculum.step()

    # Convert data to tensors
    data_tensor = torch.tensor(next_input, dtype=Conf.dtype, device=Conf.dev)
    target_tensor = torch.tensor(next_target, dtype=Conf.dtype, device=Conf.dev)

    for trial in range(num_trials):
        """Loop through trials (chunks of 4 timesteps)."""
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

        # loss_trial = criterion(logits[:, -1, :], target_tensor[:, -1, :])

        logits_ = torch.transpose(logits, 1, 2)
        targets_ = torch.transpose(target_tensor, 1, 2)
        # loss_trial = criterion(logits_[:, :, -2:], targets_[:, :, -2:])
        loss_trial = criterion(logits_, targets_)
        loss += loss_trial

        # Prepare next input based on the output and target (computes if reward should be recieved, 
        # and also whether reversal or block switch occurs)
        next_input, next_target, ground_truth = data_curriculum.step(choices.cpu().detach().numpy(), 
                                                          ground_truth)

        data_tensor = torch.tensor(next_input, dtype=Conf.dtype, device=Conf.dev)
        target_tensor = torch.tensor(next_target, dtype=Conf.dtype, device=Conf.dev)
    
    return loss


for epoch in range(Conf.num_epochs):
    # empty logger data, put model in train mode, reset optimizer
    data_curriculum.reset(train=True)
    check_train_test_split(data_curriculum, train=True)

    logger.reset()
    model.train()
    optimizer.zero_grad()

    start = time()
    loss = step(model, Conf.num_trials, logger)
    forward_time = time() - start
    if not debug:
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    end = time()
    backward_time = end - (start + forward_time)
    epoch_time = (end - start) if epoch == 0 else epoch_time + (end - start) / Conf.print_loss_interval

    # Print loss
    if epoch % Conf.print_loss_interval == 0:
        model.eval()
        logger.reset()
        data_curriculum.reset(train=False)
        check_train_test_split(data_curriculum, train=False)

        with torch.no_grad():
            step(model, Conf.num_trials_test, logger)

        print(f"\nEpoch [{epoch}/{Conf.num_epochs}]")
        print(f"Time:\t\t\t{epoch_time:.4f}s")
        print(f"F-Time:\t\t\t{forward_time:.4f}s")
        print(f"B-Time:\t\t\t{backward_time:.4f}s")
        print(f"Loss:\t\t\t{loss.item():.4f}")
        logger.get_data()
        logger.print()
        epoch_time = 0

        # data_curriculum.reset(train=True)
        # check_train_test_split(data_curriculum, train=True)

        # data_curriculum.reset(train=True)

        if epoch % Conf.save_data_interval == 0:
            logger.save_data(fname='data_' + str(epoch))

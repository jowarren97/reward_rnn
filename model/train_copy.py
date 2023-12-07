"""
X figure out dev stuff
- make a trainer class
- run on cluster
X no stim input onehot !!! should get cleaner reps
- save train/test split
- logging + tensorboard logging; save logits not choices
- function to move between A/B binary and one-hot representations
- mechanism to remove block switching (time with epoch starts only)
- general formulation of rnn: one for standard, one for groupRNN
- better logger: want to log model, environment, and curriculum, training params
- checkpointing of models
- save scripts?
- curriculum for dropout annealing?
"""

from curriculum import check_train_test_split, train_test_split
from model import SimpleRNN
import torch.nn as nn
import torch
from config import Conf
from environment import ReversalEnvironment
from logger import LearningLogger
from time import time
import os

print("Using device: ", Conf.dev)

debug = False
if debug:
    Conf.batch_size = 1

# Initialize model, curriculum, loss function and optimizer
load_dir = os.path.join(os.getcwd(), Conf.save_dir)
model = SimpleRNN(Conf)
model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_4000.pth')))

train_layouts, test_layouts, _ = train_test_split()
train_env, test_env = ReversalEnvironment(Conf, train_layouts), ReversalEnvironment(Conf, test_layouts)
logger = LearningLogger(Conf)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Conf.lr)
epoch_time = 0

def compute_weight_reg_loss(model, lambda_reg=1e-8, type='l2'):
    reg = torch.tensor(0.).to(Conf.dev)
    for param in model.parameters():
        reg += torch.norm(param, p=1 if type=='l1' else 2)
    return lambda_reg * reg

def compute_activity_reg_loss(hiddens, lambda_reg=3e-4, type='l2'):
    reg = torch.tensor(0.).to(Conf.dev)
    reg = torch.norm(hiddens, p=1 if type=='l1' else 2)
    return lambda_reg * reg

def step(model, num_trials, logger=None):
    """Do one epoch."""
    hidden = None
    loss_w_reg, loss_h_reg = 0, 0
    # Get the initial sequence for the epoch
    with torch.no_grad():
        start = time()
        inputs, targets, groundtruths = train_env.get_batch(num_trials, dropout=0.8)
        # Convert data to tensors
        data_tensor = inputs.to(dtype=Conf.dtype, device=Conf.dev)
        target_tensor = targets.to(dtype=Conf.dtype, device=Conf.dev)
        # print(f'data time {time()-start:.4f}')

    """Loop through trials (chunks of 4 timesteps)."""
    # Forward pass
    start = time()
    logits, hidden, hiddens = model(data_tensor, hidden)
    # print(f'fwd time {time()-start:.4f}')

    # store data in logger for later computation of accuracies
    if logger is not None:
        logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
                    groundtruths.cpu().detach(), train_env.optimal_agent.p_A_high.cpu().detach(), hiddens.cpu().detach())

    # loss_trial = criterion(logits[:, -1, :], target_tensor[:, -1, :])

    logits_ = torch.transpose(logits, 1, 2)
    targets_ = torch.transpose(target_tensor, 1, 2)
    # loss_trial = criterion(logits_[:, :, -2:], targets_[:, :, -2:])
    logits_state, _, logits_action = train_env.split_data(logits_, dim=1)
    targets_state, _, targets_action = train_env.split_data(targets_, dim=1)
    loss_state = criterion(logits_state, targets_state)
    loss_action = criterion(logits_action, targets_action)

    if Conf.weight_regularization > 0:
        loss_w_reg = compute_weight_reg_loss(model, lambda_reg=Conf.weight_regularization, type='l2')
    if Conf.activity_regularization > 0:
        loss_h_reg = compute_activity_reg_loss(hiddens, lambda_reg=Conf.activity_regularization, type='l2')

    if Conf.predict_x:
        loss = loss_state + loss_action
    else:
        loss = loss_action
        # print(f'l_a: {loss_trials_action:.2f}')

    return loss, loss_w_reg, loss_h_reg

# def step_offline(model, num_trials, logger=None):
#     """Do one epoch."""
#     hidden = None
#     loss_trials, loss_w_regs, loss_h_regs = 0, 0, 0
#     # Get the initial sequence for the epoch
#     next_input, next_target, ground_truth = data_curriculum.get_offline_data_sequence(num_trials)
#     # Convert data to tensors
#     data_tensor = torch.tensor(next_input, dtype=Conf.dtype, device=Conf.dev)
#     target_tensor = torch.tensor(next_target, dtype=Conf.dtype, device=Conf.dev)
    
#     logits, hidden, hiddens = model(data_tensor, hidden)

#     # store data in logger for later computation of accuracies
#     if logger is not None:
#         logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
#                     ground_truth, data_curriculum.optimal_agent.p_A_high, hiddens.cpu().detach())

#     logits_ = torch.transpose(logits, 1, 2)
#     targets_ = torch.transpose(target_tensor, 1, 2)
#     # loss_trial = criterion(logits_[:, :, -2:], targets_[:, :, -2:])
#     loss_trial = criterion(logits_, targets_)
#     loss_trials += loss_trial
#     if Conf.weight_regularization > 0:
#         loss_w_reg = compute_weight_reg_loss(model, lambda_reg=Conf.weight_regularization, type='l2')
#         loss_w_regs += loss_w_reg
#     if Conf.activity_regularization > 0:
#         loss_h_reg = compute_activity_reg_loss(hiddens, lambda_reg=Conf.activity_regularization, type='l2')
#         loss_h_regs += loss_h_reg

#     return loss_trials, loss_w_regs, loss_h_regs

for epoch in range(Conf.num_epochs):
    # empty logger data, put model in train mode, reset optimizer
    logger.reset()
    model.train()
    optimizer.zero_grad()

    start = time()
    # loss_trials, loss_w_regs, loss_h_regs = step_offline(model, Conf.num_trials, logger)
    loss_trials, loss_w_regs, loss_h_regs = step(model, Conf.num_trials, logger)
    forward_time = time() - start
    if not debug:
        # Backward pass and optimization
        loss = loss_trials + loss_w_regs + loss_h_regs
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

        with torch.no_grad():
            step(model, Conf.num_trials_test, logger)

        logger.get_data()
        logger.print()
        epoch_time = 0

        print(f"\nEpoch [{epoch}/{Conf.num_epochs}]")
        print(f"Time:\t\t\t{epoch_time:.4f}s")
        print(f"F-Time:\t\t\t{forward_time:.4f}s")
        print(f"B-Time:\t\t\t{backward_time:.4f}s")
        print(f"Loss:\t\t\t{loss.item():.4f}")
        print(f"Loss t:\t\t\t{loss_trials.item():.4f}")
        if Conf.weight_regularization > 0: print(f"Loss w:\t\t\t{loss_w_regs.item():.4f}")
        if Conf.activity_regularization > 0: print(f"Loss h:\t\t\t{loss_h_regs.item():.4f}")
        # data_curriculum.reset(train=True)
        # check_train_test_split(data_curriculum, train=True)

        # data_curriculum.reset(train=True)

        if epoch % Conf.save_data_interval == 0:
        #     logger.save_data(fname='data_' + str(epoch))
            torch.save(model.state_dict(), os.path.join(logger.save_dir, 'weights_' + str(epoch) + '.pth'))
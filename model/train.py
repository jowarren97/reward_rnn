"""
NOTES:
- current data method:
    - feed in 4 timesteps at a time to RNN: reward, ITI, init, choice
    - reward is computed using RNN choice from previous timestep
    - RNN outputs actions: do nothing, do nothing, port, port
    - targets are the perfect actions (no bayesian yet)
    - gets accuracy on each step: 1.0, 1.0, 1.0, 0.5
    - for final step (choice, just picks same port regardless of reversals)

- do I need to change the data method?
    - alternative is to compute behaviour offline using ideal bayesian agent

- make sure feedfwd input ~ recurrent input, can scale initialisation of fwd pathway
- hold out some layouts? yes, do train test error
- param norms, diff losses, act norms
"""

from curriculum import DataCurriculum
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
logger = LearningLogger()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = Conf.lr)
epoch_time = 0

def step(model, num_trials, logger=None):
    hidden = None
    loss = 0
    # Get the initial sequence for the epoch
    data_sequence = data_curriculum.get_data_sequence()
    target_sequence = data_curriculum.get_target_sequence()

    # Convert data to tensors
    data_tensor = torch.tensor(data_sequence, dtype=torch.float32, device=Conf.dev)
    target_tensor = torch.tensor(target_sequence, dtype=torch.float32, device=Conf.dev)

    for trial in range(num_trials):
        # Forward pass
        logits, hidden = model(data_tensor, hidden)

        if Conf.sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            choices = torch.nn.functional.one_hot(dist.sample(), num_classes=Conf.input_dim)
        else:
            choices = logits
        
        # store data in logger for later computation of accuracies
        if logger is not None:
            logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach())

        loss_trial = criterion(logits, target_tensor)
        loss += loss_trial

        # Prepare next input based on the output and target (computes if reward should be recieved, 
        # and also whether reversal or block switch occurs)
        next_input = data_curriculum.set_and_get_next_input(choices.cpu().detach().numpy(), 
                                                            target_tensor.cpu().detach().numpy())
        next_target = data_curriculum.get_target_sequence()

        data_tensor = torch.tensor(next_input, dtype=torch.float32, device=Conf.dev)
        target_tensor = torch.tensor(next_target, dtype=torch.float32, device=Conf.dev)
    
    return loss

for epoch in range(Conf.num_epochs):
    # empty logger data, put model in train mode, reset optimizer
    logger.reset()
    model.train()
    optimizer.zero_grad()

    start = time()
    loss = step(model, Conf.num_trials, logger)

    if not debug:
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    end = time()
    epoch_time = (end - start) if epoch == 0 else epoch_time + (end - start) / Conf.print_loss_interval

    # Print loss
    if epoch % Conf.print_loss_interval == 0:
        model.eval()
        logger.reset()

        with torch.no_grad():
            step(model, Conf.num_trials_test, logger)
        
        # print('accuracy all steps:', logger.accuracy_all.numpy())
        # print('accuracy each step:', logger.accuracy_steps)
        # print('accuracy A or B:', logger.accuracy_pair.numpy())
        # print('accuracy steps 1:', acc_1, ', accuracy_steps 2:', acc_2)
        print(f"\nEpoch [{epoch}/{Conf.num_epochs}]")
        print(f"Time:\t\t\t{epoch_time:.2f}s")
        print(f"Loss:\t\t\t{loss.item():.4f}")
        logger.get_data()
        logger.print()
        epoch_time = 0

        if epoch % Conf.save_data_interval == 0:
            logger.save_data(fname = Conf.save_dir + 'data_' + str(epoch))
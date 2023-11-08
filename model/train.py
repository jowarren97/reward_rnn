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
        
        # store data in logger for later computation of accuracies
        if logger is not None:
            logger.log(logits.detach(), target_tensor.detach(), data_tensor.detach())

        loss_trial = criterion(logits, target_tensor)
        loss += loss_trial

        # Prepare next input based on the output and target (computes if reward should be recieved, 
        # and also whether reversal or block switch occurs)
        next_input = data_curriculum.set_and_get_next_input(logits.cpu().detach().numpy(), 
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

    loss = step(model, Conf.num_trials, logger)

    if not debug:
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Print loss
    if epoch % Conf.print_loss_interval == 0:
        model.eval()
        logger.reset()

        with torch.no_grad():
            step(model, Conf.num_trials_test, logger)
        
        logger.get_data()
        print('accuracy all:', logger.accuracy_all.numpy())
        print('accuracy steps:', logger.accuracy_steps, ', pair:', logger.accuracy_pair.numpy())
        # print('accuracy steps 1:', acc_1, ', accuracy_steps 2:', acc_2)
        print(f"Epoch [{epoch}/{Conf.num_epochs}], Loss: {loss.item():.4f}")

        if epoch % Conf.save_data_interval == 0:
            logger.save_data(fname = Conf.save_dir + 'data_' + str(epoch))
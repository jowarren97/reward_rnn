"""
To try:
- smaller blocks (force less prediction based on previous action)

IN
scp -r jwarren@ssh.swc.ucl.ac.uk:./notebooks_paper/model/ /Users/jo/notebooks_paper

OUT
scp -r /Users/jo/notebooks_paper/model/ jwarren@ssh.swc.ucl.ac.uk:./notebooks_paper/


X figure out dev stuff
X make a trainer class
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
from model import SimpleRNN
from config import Conf
from trainer import ReversalTrainer
from scheduler import ReversalScheduler
import os
import torch


config = Conf()

print("Using device: ", config.dev)

debug = False
if debug:
    config.batch_size = 1

# Initialize model, curriculum, loss function and optimizer
load_dir = os.path.join(os.getcwd(), config.save_dir)
model = SimpleRNN(config)
# model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_4000.pth')))

optimizer_func = torch.optim.AdamW
optimizer_kwargs = {"lr": config.lr, "weight_decay": config.weight_regularization, "amsgrad": config.amsgrad}
scheduler_func = torch.optim.lr_scheduler.LinearLR
scheduler_kwargs = {"start_factor": 1.0, "end_factor": 0.1, "total_iters": config.decay_epochs}
env_scheduler = ReversalScheduler(total_iters=config.dropout_decay_epochs)

trainer = ReversalTrainer(model, 
                          config, 
                          optimizer_func, 
                          scheduler_func,
                          env_scheduler=env_scheduler, 
                          optimizer_kwargs=optimizer_kwargs, 
                          scheduler_kwargs=scheduler_kwargs)

trainer.train()
"""
IN
scp -r jwarren@ssh.swc.ucl.ac.uk:./notebooks_paper/model/ /Users/jo/notebooks_paper

OUT
scp -r /Users/jo/notebooks_paper/model/ jwarren@ssh.swc.ucl.ac.uk:./notebooks_paper/

scp -r jwarren@ssh.swc.ucl.ac.uk:./notebooks_paper/run_data/2024-01-23/cd6e787/p_0.8_lr_1e-04-1e-05_batchsize_64_h_256_wreg_1e-06_hreg_3e-04_thresh_None_wgain_5e-01_lrdecay_10000_dropoutdecay_1000 /Users/jo/notebooks_paper

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

print("Using device: ", Conf.dev)

debug = False
if debug:
    Conf.batch_size = 1

# Initialize model, curriculum, loss function and optimizer
load_dir = os.path.join(os.getcwd(), Conf.save_dir)
model = SimpleRNN(Conf)
# model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_4000.pth')))

optimizer_func = torch.optim.AdamW
optimizer_kwargs = {"lr": Conf.lr, "weight_decay": Conf.weight_regularization, "amsgrad": Conf.amsgrad}
scheduler_func = torch.optim.lr_scheduler.LinearLR
scheduler_kwargs = {"start_factor": 1.0, "end_factor": 0.1, "total_iters": Conf.decay_epochs}
env_scheduler = ReversalScheduler(total_iters=Conf.dropout_decay_epochs)

trainer = ReversalTrainer(model, 
                          Conf, 
                          optimizer_func, 
                          scheduler_func,
                          env_scheduler=env_scheduler, 
                          optimizer_kwargs=optimizer_kwargs, 
                          scheduler_kwargs=scheduler_kwargs)

trainer.train()
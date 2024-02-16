from environment import ReversalEnvironment, train_test_split
from model import SimpleRNN
import torch
from config import Conf
from logger_old import LearningLogger
from time import time
import os
import numpy as np

print("Using device: ", Conf.dev)

Conf.batch_size = 10000
# Conf.hidden_dim = 128
Conf.num_epochs_test = 1
print(Conf.num_trials_test)
# load_dir = 'run_data_08_256_l2_1e6_l2_3e4'
# load_dir = 'run_data'
load_dir = '/Users/jo/notebooks_paper/run_data/2024-02-05/Untitled/18:37'
# load_dir = '/Users/jo/notebooks_paper/run_data/2024-01-16/5c363ba/p_0.8_lr_1e-04-1e-05_batchsize_64_h_256_wreg_1e-06_hreg_3e-04_thresh_None_wgain_5e-01_lrdecay_10000_dropoutdecay_1000/16:27/'
# load_dir = '/Users/jo/notebooks_paper/model/run_data/2023-12-12/cd6e787/p_0.8_lr_1e-04_batchsize_64_h_256_wreg_1e-06_hreg_3e-04_thresh_None_wgain_2e-01/11:50/'
# Conf.save_dir = os.path.join(os.getcwd(), 'run_data_remote')
Conf.save_dir = load_dir

# Initialize model, curriculum, loss function and optimizer
model = SimpleRNN(Conf)
model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_1050.pth'), map_location=torch.device(Conf.dev)))
data = np.load(os.path.join(load_dir, 'train_test_split.npz'))
train_layouts, test_layouts, all_layouts = data['train'], data['test'], data['all']
print('len', len(test_layouts))
# train_layouts, test_layouts, all_layouts = train_test_split()
train_env, test_env, all_env = ReversalEnvironment(Conf, train_layouts), ReversalEnvironment(Conf, test_layouts), ReversalEnvironment(Conf, all_layouts)

logger = LearningLogger(Conf)

def step(model, env, num_trials, logger=None):
    """Do one epoch."""
    hidden = None
    print('step')
    # Get the initial sequence for the epoch
    with torch.no_grad():
        inputs, targets, groundtruths = env.get_batch(num_trials, {'dropout': 0.0})
        # Convert data to tensors
        data_tensor = inputs.to(dtype=Conf.dtype, device=Conf.dev)
        target_tensor = targets.to(dtype=Conf.dtype, device=Conf.dev)

    """Loop through trials (chunks of 4 timesteps)."""
    # Forward pass
    logits, hidden, hiddens = model(data_tensor, hidden)

    # store data in logger for later computation of accuracies
    if logger is not None:
        logger.log(logits.cpu().detach(), target_tensor.cpu().detach(), data_tensor.cpu().detach(), 
                    groundtruths, torch.cat(env.log['p_A_high']['data']).cpu().detach(), hiddens.cpu().detach())

    return


model.eval()

for i in range(Conf.num_epochs_test):
    print('TRAIN')
    logger.reset()
    step(model, train_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname=f'data_train')
    logger.print()

    print('TEST')
    logger.reset()
    step(model, test_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname=f'data_test')
    logger.print()

    print('ALL')
    logger.reset()
    step(model, all_env, num_trials=Conf.num_trials_test, logger=logger)
    logger.get_data()
    logger.save_data(fname=f'data_all')
    logger.print()

    # logger.reset_logs()
    # step(model, train_env, num_trials=Conf.num_trials_test)
    # logger.save_data()

    # logger.reset_logs()
    # step(model, test_env, num_trials=Conf.num_trials_test)
    # logger.save_data()

    # logger.reset_logs()
    # step(model, all_env, num_trials=Conf.num_trials_test)
    # logger.save_data()

# logger.save_data(fname='data_all')

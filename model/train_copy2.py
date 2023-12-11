"""
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

from curriculum import check_train_test_split, train_test_split
from model import SimpleRNN
from config import Conf
from trainer import ReversalTrainer
import os

print("Using device: ", Conf.dev)

debug = False
if debug:
    Conf.batch_size = 1

# Initialize model, curriculum, loss function and optimizer
load_dir = os.path.join(os.getcwd(), Conf.save_dir)
model = SimpleRNN(Conf)
# model.load_state_dict(torch.load(os.path.join(load_dir, 'weights_4000.pth')))

trainer = ReversalTrainer(model, Conf)

trainer.train()
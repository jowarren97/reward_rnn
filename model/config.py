import torch

class Conf(): 
    # if torch.backends.mps.is_available():
    #     dev = torch.device("mps")
    # else:
    #     dev = torch.device("cpu")
    dev = torch.device("cpu")

    lr = 0.001
    num_epochs = 5000
    num_trials = 100  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = 1000

    reward_prob = 1.0
    n_reversals = 4
    mean_trials_since_reversal = 15
    jitter = 5

    batch_size = 64
    input_dim = 10  # onehot: 9 port lights, 1 reward input
    hidden_dim = 200
    output_dim = 10  # onehot: 9 port choices, 1 do-nothing choice
    threshold = 5.0  # choose None to use vanilla RNN

    print_loss_interval = 50
    save_data_interval = 100
    save_dir = '../'



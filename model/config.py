import torch
import os
import os


class Conf():
    dtype = torch.float32
    dtype = torch.float32
    dev = torch.device("cpu")
    # # slower with mps for some reason
    # if torch.backends.mps.is_available():
    #     dev = torch.device("mps")
    # else:
    #     dev = torch.device("cpu")

    # curriculum params
    lr = 0.0001
    num_epochs = 100000
    trial_len = 5
    num_trials = 50  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = 1000
    batch_size = 64

    # network params
    port_dim = 9
    input_dim = port_dim + 2  # onehot: 9 port lights, 1 reward input
    hidden_dim = 256
    output_dim = port_dim + 1  # onehot: 9 port choices, 1 do-nothing choice

    # experimental stuff
    sample = False  # whether to sample action from RNN logits, or take max logit as action 
    # (maybe if sample then RNN should receive action taken?)
    weight_init = True  # whether to use weight inits on RNN
    threshold = 5.0  # choose None to use vanilla RNN
    use_rnn_actions = False

    # task params
    reward_prob = 1.0
    n_reversals = 10
    max_trials_since_reversal = 8
    p_switch = 1/max_trials_since_reversal
    jitter = 1  # jitter in reward probability

    # data output params
    print_loss_interval = 50
    save_data_interval = 1000
    save_dir = os.path.join(os.getcwd(), 'run_data')

import torch

class Conf():
    dev = torch.device("cpu")
    # # slower with mps for some reason
    # if torch.backends.mps.is_available():
    #     dev = torch.device("mps")
    # else:
    #     dev = torch.device("cpu")
    
    # curriculum params
    lr = 0.001
    num_epochs = 5000
    num_trials = 100  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = 1000
    batch_size = 64

    # network params
    input_dim = 10  # onehot: 9 port lights, 1 reward input
    hidden_dim = 200
    output_dim = 10  # onehot: 9 port choices, 1 do-nothing choice

    # experimental stuff
    sample = False  # whether to sample action from RNN logits, or take max logit as action 
                   # (maybe if sample then RNN should receive action taken?)
    weight_init = False  # whether to use weight inits on RNN
    threshold = 5.0  # choose None to use vanilla RNN

    # task params
    reward_prob = 1.0
    n_reversals = 10
    max_trials_since_reversal = 15
    jitter = 5

    # data output params
    print_loss_interval = 50
    save_data_interval = 100
    save_dir = '../'



import torch
import os
import os


class ConfSlot():
    dtype = torch.float32
    # dev = torch.device("cpu")
    # slower with mps for some reason
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # curriculum params
    lr = 1e-4
    num_epochs = 10000
    num_trials = 5  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = num_trials
    batch_size = 128

    # trial params
    trial_len = 2
    # trial_len = 5
    port_dim = 50
    x_t_mask = [1 for _ in range(trial_len)]
    r_t_mask = [0 for _ in range(trial_len)]
    a_t_mask = [0 for _ in range(trial_len)]

    hidden_dim = 256

    # experimental stuff
    sample = False  # whether to sample action from RNN logits, or take max logit as action 
    # (maybe if sample then RNN should receive action taken?)
    weight_init = True  # whether to use weight inits on RNN
    threshold = None #5.0  # choose None to use vanilla RNN
    use_rnn_actions = False
    weight_regularization = 1e-6 #1e-7  # weight regularization
    activity_regularization = 1e-3 #1e-4
    predict_x = True

    # data output params
    print_loss_interval = 50
    save_data_interval = 1000
    save_dir = os.path.join(os.getcwd(), 'run_data_new')

    a_dim = 0
    x_dim = port_dim
    r_dim = 0
    t_ax = 1
    batch_ax = 0
    env_dev = 'cpu'
    no_stim_token, no_reward_token, no_action_token = True, False, False
    if no_stim_token: x_dim += 1
    if no_reward_token: r_dim += 1
    if no_action_token: a_dim += 1

    input_dim = x_dim + r_dim + a_dim  # onehot: 9 port lights, 2 reward input
    output_dim = x_dim + r_dim + a_dim

    save_type = "SAVE_DICT"  # "SAVE_OBJECT"
    n_batches = 50
    save_model_interval = 50
    init_hh_weight_gain = 0.5

    start_lr = 1e-4
    end_lr = 1e-5
    decay_epochs = 10000
    opt = 'AdamW'
    amsgrad = False
    # dropout = 0.9
    loss_trial_start = 1
    target_shift = -1

    dropout_decay_epochs = 1000

    def get_path(self):
        return ''
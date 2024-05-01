import torch
import os
import os

"""
657 port100 provide_action true
658 port100 provide_action false
659 port50 provide action false
660 port50 provide action true

661 port50 w0                           0.008, 0.07, 0.011
663 port50 w1                           0.001, 0.07, 0.007
664 port100 w1                          0.002, 0.07, 0.012
665 port200 w1 h1024                    0.002, 0.003, 0.018 ***
666 port9 w1                            X
668 port50 w1 h1024?                    0.007, 0.08, 0.008
692 port100 w1 h1024?                   0.001, 0.02, 0.011
696 port100 w1 512 provide_action f     0.01,  0.02, 0.03

4555696 port100 w1 h512 provide_action false
"""


class Conf():
    dtype = torch.float32
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
    num_trials = 40  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = num_trials
    num_epochs_test = 10
    batch_size = 128
    n_batches = 50

    # trial params
    trial_len = 2
    pre_trial = True
    # trial_len = 5
    port_dim = 200
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 2, 5, 6, 3, 7
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, None, 1, 2, None, 3
    r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, None, None, None, None, 1
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 1, 4, 5, 2, 6
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 2, 3, 4, 2, 4
    x_t_mask = [0 for _ in range(trial_len)]
    r_t_mask = [0 for _ in range(trial_len)]
    a_t_mask = [1 for _ in range(trial_len)]
    a_t_mask[-1] = 1
    if ab_choice_step >= trial_len:
        raise ValueError("ab_choice_step must be less than trial_len")
    state_dim = port_dim + 2
    action_dim = port_dim + 1  # onehot: 9 port choices, 1 do-nothing choice
    input_dim = state_dim + action_dim  # onehot: 9 port lights, 2 reward input
    hidden_dim = 1024

    # experimental stuff
    sample = False  # whether to sample action from RNN logits, or take max logit as action 
    # (maybe if sample then RNN should receive action taken?)
    weight_init = True  # whether to use weight inits on RNN
    threshold = None #5.0  # choose None to use vanilla RNN
    use_rnn_actions = False
    weight_regularization = 1e-6 #1e-7  # weight regularization
    activity_regularization = 3e-4 #1e-4
    predict_x = True
    provide_actions = False
    output_dim = input_dim if predict_x else action_dim

    # task params
    reward_prob = 0.7
    n_reversals = 10000000
    max_trials_since_reversal = 6
    p_switch = 1/max_trials_since_reversal
    jitter = 3  # jitter in reward probability

    # data output params
    print_loss_interval = 50
    save_data_interval = 1000
    save_dir = os.path.join(os.getcwd(), 'run_data_new')

    a_dim = port_dim
    x_dim = port_dim
    r_dim = 2
    t_ax = 1
    batch_ax = 0
    env_dev = 'cpu'
    no_stim_token, no_reward_token, no_action_token = True, False, True
    if no_stim_token: x_dim += 1
    if no_reward_token: r_dim += 1
    if no_action_token: a_dim += 1

    input_dim = x_dim + r_dim + a_dim  # onehot: 9 port lights, 2 reward input
    output_dim = x_dim + r_dim + a_dim

    save_type = "SAVE_DICT"  # "SAVE_OBJECT"
    save_model_interval = 50
    init_hh_weight_gain = 1.0

    start_lr = 1e-4
    end_lr = 1e-5
    decay_epochs = 10000
    opt = 'AdamW'
    amsgrad = False
    # dropout = 0.9
    loss_step_start = 2 if a_step is None and b_step is None else trial_len

    dropout_decay_epochs = 1000

    def get_path(self):
        return ''
    


    """
    import torch
import os
import os


class Conf():
    dtype = torch.float32
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
    num_trials = 40  # per epoch; timesteps = num_trials * 4  (reward, delay, init, choice)
    num_trials_test = num_trials
    num_epochs_test = 10
    batch_size = 256

    # trial params
    trial_len = 4
    # trial_len = 5
    port_dim = 50
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 2, 5, 6, 3, 7
    r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, None, 1, 2, None, 3
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 1, 4, 5, 2, 6
    # r_step, init_step, a_step, b_step, init_choice_step, ab_choice_step = 0, 2, 3, 4, 2, 4
    x_t_mask = [0, 0, 0, 0]
    r_t_mask = [0, 0, 0, 0]
    a_t_mask = [0, 0, 0, 1]
    assert ab_choice_step < trial_len
    assert a_step != b_step != init_step != r_step
    state_dim = port_dim + 2
    action_dim = port_dim + 1  # onehot: 9 port choices, 1 do-nothing choice
    input_dim = state_dim + action_dim  # onehot: 9 port lights, 2 reward input
    hidden_dim = 512

    # experimental stuff
    sample = False  # whether to sample action from RNN logits, or take max logit as action 
    # (maybe if sample then RNN should receive action taken?)
    weight_init = True  # whether to use weight inits on RNN
    threshold = None #5.0  # choose None to use vanilla RNN
    use_rnn_actions = False
    weight_regularization = 1e-6 #1e-7  # weight regularization
    activity_regularization = 3e-4 #1e-4
    predict_x = True
    output_dim = input_dim if predict_x else action_dim

    # task params
    reward_prob = 0.8
    n_reversals = 10000000
    max_trials_since_reversal = 5
    p_switch = 1/max_trials_since_reversal
    jitter = 3  # jitter in reward probability

    # data output params
    print_loss_interval = 50
    save_data_interval = 1000
    save_dir = os.path.join(os.getcwd(), 'run_data_new')

    a_dim = port_dim
    x_dim = port_dim
    r_dim = 2
    t_ax = 1
    batch_ax = 0
    env_dev = 'cpu'
    no_stim_token, no_reward_token, no_action_token = True, False, True
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

    dropout_decay_epochs = 1000

    def get_path(self):
        return ''
        """
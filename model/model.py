import torch
import torch.nn as nn
from config import Conf


def init_weights_thresholded_rnn(rnn_cell):
    nn.init.xavier_uniform_(rnn_cell.i2h.weight)
    nn.init.trunc_normal_(rnn_cell.h2h.weight, mean=0.0, std=0.001)
    # nn.init.eye_(rnn_cell.h2h.weight)
    # rnn_cell.h2h.weight.data += torch.diag(torch.normal(torch.zeros(rnn_cell.h2h.weight.size(0)), 0.01))
    rnn_cell.i2h.bias.data.zero_()
    rnn_cell.h2h.bias.data.zero_()

def init_weights_rnn(rnn, gain):
    for name, param in rnn.named_parameters():
        # if 'weight_ih' in name:  # Input-to-hidden weights
        # print(param.data)
        #     nn.init.xavier_uniform_(param.data)
        if 'weight' in name:
            if 'hh' in name:  # Hidden-to-hidden weights
            # nn.init.eye_(param.data) + torch.normal(torch.zeros_like(param.data), 0.01, dtype=Conf.dtype, device=Conf.dev)
                nn.init.trunc_normal_(param.data, mean=0.0, std=0.001)
                param.data += gain*torch.eye(param.size(0)).to(param.device)
            elif 'ih' in name:
                nn.init.xavier_uniform_(param.data)
        if 'bias' in name:
            param.data.zero_()

        # elif 'bias' in name:
        #     param.data.zero_()


class ThresholdedRNNCell(nn.Module):
    """Had to make custom RNN cell bc torch.nn.rnn.forward() processes multiple timesteps (want to threshold 
    act each timestep)"""

    def __init__(self, input_size, hidden_size, output_size, threshold, device, dtype):
        super(ThresholdedRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.dtype = dtype
        self.i2h = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.h2h = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.device = device

        self.hidden_init = nn.Parameter(torch.zeros((1, self.hidden_size), dtype=dtype, device=device),
                                        requires_grad=True)

    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.hidden_init.tile([x.size(0), 1])
            # hidden = torch.zeros(x.size(0), self.hidden_size, device=self.device, dtype=self.dtype)

        outputs = []
        # Process each time step with the RNN cell
        for t in range(x.size(1)):
            hidden = self._step(x[:, t], hidden)
            # out = self.h2a(hidden)
            # outputs.append(out)
            outputs.append(hidden)

        rnn_out = torch.stack(outputs, dim=1)
        return rnn_out, hidden

    def _step(self, x, hidden):
        pre_activation = hidden + self.i2h(x) + self.h2h(hidden)
        # Apply thresholded ReLU
        act = torch.nn.functional.relu(pre_activation)
        act = torch.minimum(act, torch.tensor(self.threshold).to(self.device))
        return act


class SimpleRNN(nn.Module):
    def __init__(self, config):
        super(SimpleRNN, self).__init__()
        self.device = config.dev

        self.log = {"hidden"        : {"data": [], "tb": False, "save": True},
                    "logits"        : {"data": [], "tb": False, "save": True},
                    "hidden_norm"   : {"data": [], "tb": True,  "save": False},
                    "hidden_max_all": {"data": [], "tb": True,  "save": False},
                    "hidden_max_mean": {"data": [], "tb": True,  "save": False},
                    "h2h_norm"      : {"data": [], "tb": True,  "save": False},
                    "i2h_norm"      : {"data": [], "tb": True,  "save": False},
                    "h2h_i2h_ratio" : {"data": [], "tb": True,  "save": False}}
        
        if config.threshold is not None:
            print('Using thresholded RNN')
            self.rnn = ThresholdedRNNCell(config.input_dim, config.hidden_dim, config.output_dim,
                                          threshold=config.threshold, device=config.dev, dtype=config.dtype)
            # weight init doesn't seem to be working
            if Conf.weight_init:
                init_weights_thresholded_rnn(self.rnn)
        else:
            print('Using standard RNN')
            self.rnn = nn.RNN(input_size=config.input_dim, hidden_size=config.hidden_dim, num_layers=1,
                              batch_first=True, nonlinearity='relu', device=config.dev, dtype=config.dtype)
            if Conf.weight_init:
                init_weights_rnn(self.rnn, config.init_hh_weight_gain)

        self.linear = nn.Linear(config.hidden_dim, config.output_dim, device=config.dev, dtype=config.dtype)

    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.linear(rnn_out)

        # self.log["hiddens"]["data"].append(rnn_out)
        # self.log["logits"]["data"].append(output)
        with torch.no_grad():
            self.log["hidden_norm"]["data"].append(torch.norm(rnn_out))
            self.log["hidden_max_mean"]["data"].append(torch.mean(torch.amax(rnn_out, dim=(0,1))))
            self.log["hidden_max_all"]["data"].append(torch.max(rnn_out))

            hh_w = self.rnn.h2h.weight if isinstance(self.rnn, ThresholdedRNNCell) else self.rnn.weight_hh_l0
            ih_w = self.rnn.i2h.weight if isinstance(self.rnn, ThresholdedRNNCell) else self.rnn.weight_ih_l0
            hh_b = self.rnn.i2h.bias if isinstance(self.rnn, ThresholdedRNNCell) else self.rnn.bias_hh_l0
            ih_b = self.rnn.i2h.bias if isinstance(self.rnn, ThresholdedRNNCell) else self.rnn.bias_ih_l0

            h2h = torch.norm(rnn_out @ hh_w + hh_b)
            i2h = torch.norm(x @ ih_w.T + ih_b)
            if h2h.shape == i2h.shape:
                self.log["h2h_norm"]["data"].append(h2h)
                self.log["i2h_norm"]["data"].append(i2h)
                self.log["h2h_i2h_ratio"]["data"].append(h2h/i2h)

        return output, hidden, rnn_out
    
    def get_log(self):
        return self.log
    
    def reset_log(self):
        for key in self.log.keys():
            self.log[key]['data'] = []

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1, device=Conf.dev):
#         super(SimpleRNN, self).__init__()
#         self.rnn = nn.RNN(input_size=input_size,
#                           hidden_size=hidden_size,
#                           num_layers=num_layers,
#                           batch_first=True,
#                           nonlinearity='relu',
#                           device=device)
#         self.linear = nn.Linear(hidden_size, 
#                                 output_size, 
#                                 device=device)

#     def forward(self, x, hidden=None):
#         rnn_out, hidden = self.rnn(x, hidden)
#         output = self.linear(rnn_out)

#         return output, hidden

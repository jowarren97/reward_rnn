import torch
import torch.nn as nn
from config import Conf


def init_weights_thresholded_rnn(rnn_cell):
    nn.init.xavier_uniform_(rnn_cell.i2h.weight)
    nn.init.xavier_uniform_(rnn_cell.h2a.weight)
    nn.init.trunc_normal_(rnn_cell.h2h.weight, mean=0.0, std=0.001)
    # nn.init.eye_(rnn_cell.h2h.weight)
    # rnn_cell.h2h.weight.data += torch.diag(torch.normal(torch.zeros(rnn_cell.h2h.weight.size(0)), 0.01))
    rnn_cell.i2h.bias.data.zero_()
    rnn_cell.h2a.bias.data.zero_()
    rnn_cell.h2h.bias.data.zero_()

def init_weights_rnn(rnn):
    for name, param in rnn.named_parameters():
        # if 'weight_ih' in name:  # Input-to-hidden weights
        # print(param.data)
        #     nn.init.xavier_uniform_(param.data)
        if 'weight_hh' in name:  # Hidden-to-hidden weights
            # nn.init.eye_(param.data) + torch.normal(torch.zeros_like(param.data), 0.01, dtype=Conf.dtype, device=Conf.dev)
            nn.init.trunc_normal_(param.data, mean=0.0, std=0.001)

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
        self.h2a = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
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
                init_weights_rnn(self.rnn)

        self.linear = nn.Linear(config.hidden_dim, config.output_dim, device=config.dev, dtype=config.dtype)
        self.linear = nn.Linear(config.hidden_dim, config.output_dim, device=config.dev, dtype=config.dtype)

    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.linear(rnn_out)

        return output, hidden, rnn_out

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

# For James

> RNN model (./model) contained within repository for original paper [here](https://github.com/veronikasamborska1994/notebooks_paper). Environment can be installed using environment.yml.

## RNN set up:

- run train.py to train network, plot_rnn.ipynb to visualise inputs/targets/outputs

### Data method
- current data method:
    - feed in 4 timesteps at a time to RNN: reward, ITI, init, choice
    - reward is computed using RNN choice from previous timestep
    - RNN outputs actions: do nothing, do nothing, port, port
    - targets are the perfect actions (no bayesian yet)
    - gets accuracy on each step: 1.0, 1.0, 1.0, 0.5
    - for final step (choice), just picks same port regardless of reversals)

- **do I need to change the data method?**
    - alternative is to compute behaviour offline using ideal bayesian agent, then train RNN on this

### Things tried (but didn't help)
- identity recurrent initialisation (with and without Xavier on feedforward, small diagonal normal distr noise, biases on/off)
- thresholding on relu units
- lots of units

### To try
- make sure feedfwd input ~ recurrent input, can scale initialisation of fwd pathway
- hold out some layouts? yes, do train-test error
- param norms, diff losses, act norms

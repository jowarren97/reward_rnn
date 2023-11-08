# For James

> RNN model (in 'model' directory), contained within a clone of full repository for original paper (see [here](https://github.com/veronikasamborska1994/notebooks_paper)). Environment can be installed using environment.yml.

## RNN set up:

- run train.py to train network, plot_rnn.ipynb to visualise inputs/targets/outputs

### Data method
- current data method:
    - feed in 4 timesteps at a time to RNN: reward, ITI, init, choice
    - reward is computed using RNN choices from previous trial (final two actions, i.e. init, choice, need to be correct)
    - RNN outputs actions: do nothing, do nothing, port id, port id
    - targets are the perfect actions (no bayesian agemt yet), reward probability is 1.0 (not 0.8 from paper)
    - gets accuracy on each step: 1.0, 1.0, 1.0, 0.5
    - for final step (choice), just picks same port regardless of reversals)

- **do I need to change the data method?**
    - alternative is to compute behaviour offline using ideal bayesian agent, then train RNN on this
    - could also add some kind of exploration in action-selection? 

### Things tried (but didn't help)
- choosing actions by sampling from RNN logits vs taking max logit (current method, set using Conf.sample)
- identity recurrent initialisation (with and without Xavier on feedforward, small diagonal normal distr noise, biases on/off)
- thresholding on relu units
- lots of units

### To try
- make sure feedfwd input ~ recurrent input, can scale initialisation of fwd pathway
- hold out some layouts? yes, do train-test error
- param norms, diff losses, act norms

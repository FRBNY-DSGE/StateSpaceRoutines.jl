# Tempered Particle Filter Example

## A Linear State Space System derived from a basic DSGE model
- The tempered_particle_filter_ex.jl example file demonstrates the functionality of our Julia implementation of
the tempered particle filter on a linear state space system obtained from solving a basic New Keynesian DSGE
model with Gaussian shocks.

- The example file first loads in a dataset, us.txt, which is an 80-period dataset from 1983-Q1 to 2002-Q4
provided by Herbst and Schorfheide in their [2017 CEF Workshop](https://web.sas.upenn.edu/schorf/cef-2017-herbst-schorfheide-workshop/),
located in the Practical Exercises/MATLAB-PF zip file.

- The solution to the model is then computed using the compute_system function, which is a wrapper script for
setting up and solving the linear rational expectations model using Chris Sims' gensys algorithm. The system
matrices in this example case define the linear transition and measurement equations `Φ` and `Ψ`, and the shock distribution
`F_ϵ` in the example is a multivariate normal distribution (but as mentioned previously, these equations can be generally
non-linear and the distribution generally non-Gaussian).

- We construct a dictionary with the hyperparameters for readability purposes; however these specifications can
also be directly entered into the tempered_particle_filter function as keyword arguments, as opposed to splatting
(`...`) a dictionary into the keyword arguments section of the function call.

- The initial states are drawn from a multivariate normal distribution centered at a prior mean `s0` and
solving the discrete-time Lyapunov equation for the variance-covariance matrix of the states, `P0`. The
`initialize_state_draws` function can be used instead to produce equivalent results.

- When executing the filter, the default return value is the approximated log-likelihood value; however, if the
keyword argument `allout` is set to be true, then the filter will also return the marginal likelihoods at each
period as well as the amount of time it took each marginal likelihood to be calculated.


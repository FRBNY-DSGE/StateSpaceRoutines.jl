# Tempered Particle Filter Example Documentation

The tempered_particle_filter_ex.jl example file demonstrates the functionality of our Julia implementation of
the tempered particle filter, originally introduced and implemented by Ed Herbst and Frank Schorfheide in a recent
paper (cited in the [README](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl) in the root directory of the
StateSpaceRoutines repository), on a linear state space system obtained from solving a basic New Keynesian DSGE model
with Gaussian shocks.

```
s_{t+1} = Φ(s_t, ϵ_t)        (transition equation)
y_t     = Ψ(s_t, u_t)        (measurement equation)

ϵ_t ∼ F_ϵ(∙; θ)
u_t ∼ F_u(∙; θ), where F_u is N(0, HH), where HH is the variance matrix of the i.i.d measurement error
Cov(ϵ_t, u_t) = 0
```

In the framework of Bayesian estimation of linear DSGE models, a given set of parameters specify linear transition and
measurement equations, i.e. a linear state space system, from which the log-likelihood can be derived. But, in a
more general environment, where the state space system is non-linear and/or the shocks driving the transition dynamics
are drawn from a non-Gaussian distribution, the log-likelihood then cannot be obtained using the standard Kalman filter,
thus motivating the need for a more general filtering method.

The example file first loads in a dataset, us.txt, which is an 80-period dataset from 1983-Q1 to 2002-Q4 provided
by Ed Herbst & Frank Schorfheide in their [2017 CEF Workshop](https://web.sas.upenn.edu/schorf/cef-2017-herbst-schorfheide-workshop/), located in the Practical Exercises/MATLAB-PF zip file.

The solution to the model is then computed using the compute_system function, which is a wrapper script for
setting up and solving the linear rational expectations model using Chris Sims' gensys algorithm. The system matrices in
this example case define the linear transition equation `Φ` and the linear measurement equation `Ψ`, but these
can be two general functions. Likewise, the shock distribution `F_ϵ` in the example is a multivariate normal distribution
but can be a general instance of the Distribution type.

We construct a dictionary with the hyperparameters for readability purposes; however these specifications can also be directly entered into the tempered_particle_filter function as keyword arguments, as opposed to splatting (`...`) a dictionary into the keyword arguments section of the function call. Detailed documentation about the tuning parameters can be found by reading the docstring, i.e. by calling `?tempered_particle_filter` in the Julia REPL after loading the StateSpaceRoutines package.

The initial states are drawn by calling the `initialize_state_draws` function, which generates draws by simulating the
states: starting from an initial point `s0`, drawing shocks from `F_ϵ`, and iterating forward. This is well-defined under
some assumptions about the stationarity and ergodicity of the state's dynamics, which is achieved after an initial burn-in
period. The sample is also thinned to reduce serial correlation.

When executing the filter, the default return value is the approximated log-likelihood value; however, if the keyword
argument `allout` is set to be true, then the filter will also return the marginal likelihoods at each period as well as
the amount of time it took each marginal likelihood to be calculated.

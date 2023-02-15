# MonteCarloSummary

## Installation
```julia
using Pkg
Pkg.add("MonteCarloSummary")
```

## Description

Have Monte Carlo simulations and need a simple, efficiently computed summary? This package provides just that.
A single function, `mcsummary`, computes the bare minimum of statistical properties -- mean, Monte Carlo standard error, standard deviation, and quantiles (the granularity of which may be specified by the user).
The assumption is that irrespective of how one's Monte Carlo simulations are generated, the result is a matrix of numeric values. Perhaps the simulation index may be on the first or second dimension -- the user may specify this with the `dim` keyword argument. Given that Monte Carlo typically involves a large number of simulations (and/or high-dimensional spaces), `mcsummary` defaults to a threaded implementation, but the user may opt out of this with the `multithreaded` keyword argument.

That's it. Enjoy.


# Simulating Yang-Mills on the lattice 

```@docs
GaugeParm
YMworkspace
ztwist
```

## Gauge actions and forces

Routines to compute the gauge action.
```@docs
gauge_action
```

Routines to compute the force derived from gauge actions. 

```@docs
force_gauge
```

### Force field refresh

Algebra fields with **natural indexing** can be randomized.
```@docs
randomize!
```


## Basic observables

Some basic observable.
```@docs
plaquette
```

## HMC simulations

### Integrating the EOM

```@docs
IntrScheme
leapfrog
omf2
omf4
MD!
```

### HMC algorithm

```@docs
hamiltonian
HMC!
```





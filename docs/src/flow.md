# Gradient flow

The gradient flow equations can be integrated in two different ways:
1. Using a fixed step-size integrator. In this approach one fixes the
   step size $\epsilon$ and the links are evolved from
   $V_\mu(t)$ to $V_\mu(t +\epsilon)$ using some integration
   scheme. 
1. Using an adaptive step-size integrator. In this approach one fixes
   the tolerance $h$ and the links are evolved for a time $t_{\rm
   end}$ (i.e. from $V_\mu(t)$ to $V_\mu(t +t_{\rm end})$)
   with the condition that the maximum error while advancing is not
   larger than $h$.
   
In general adaptive step size integrators are much more efficient, but
one loses the possibility to measure flow quantities at the
intermediate times $\epsilon, 2\epsilon, 3\epsilon,...$. Adaptive
step size integrators are ideal for finite size scaling studies, while
a mix of both integrators is the most efficient approach in scale
setting applications. 

## Integration schemes

```@docs
FlowIntr
wfl_euler
zfl_euler
wfl_rk2
zfl_rk2
wfl_rk3
zfl_rk3
```

## Integrating the flow equations

```@docs
flw
flw_adapt
```

## Observables

```@docs
Eoft_plaq
Eoft_clover
Qtop
```


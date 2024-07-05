
# Solvers

The module `Solvers` contains the functions to invert the Dirac
operator as well as functions to obtain specific propagators.


## CG.jl

The function [`CG!`](@ref) implements the Conjugate gradient 
algorith for the operator A

```@docs
CG!
```

where the tolerance is normalized with respect to $$|$$``si``$$|^2$$.
The operator A must have the same input structure as all the Dirac
operators. If the maximum number of iterations `maxiter` is reached,
the function will throw an error. The estimation for $$A^{-1}x$$ is
stored in ``si``, and the number of iterations is returned.


Note that all the fermion field in ``dws`` are used
inside the function and will be modified. In particular, the final residue
is given by $$|$$``dws.sr``$$|^2$$.



## Propagators.jl

In this file, we define some useful functions to obtain certain
propagators.

```@docs
propagator!
```

Note that the indexing in Julia starts at 1, so the first tiime slice is t=1.

Internally, this function solves the equation 

```math
    D_w^\dagger D_w \psi = \gamma_5 D_w \gamma_5 \eta
```

where $$\eta$$ is either a point-source with the specified color and spin
or a random source in a time-slice and stores the value in ``pro``. 
To solve this equation, the [`CG!`](@ref) function is used.


For the case of SF boundary conditions, we have the boundary-to-bulk 
propagator, implemented by the function [`bndpropagator!`](@ref)

```@docs
bndpropagator!
```

This propagator is defined by the equation:

```math
D_W S(x) = \frac{c_t}{\sqrt{V}} \delta_{x_0,1} U_0^\dagger(0,\vec{x}) P_+
```

The analog for the other boundary is implemented in the function [`Tbndpropagator!`](@ref)

```@docs
Tbndpropagator!
```

defined by the equation:

```math
D_W R(x) = \frac{c_t}{\sqrt{V}} \delta_{x_0,T-1} U_0(T-1,\vec{x}) P_-
```

Where $$P_\pm = (1 \pm \gamma_0)/2$$. The boundary-to-boundary 
propagator

```math
\frac{-c_t}{\sqrt{V}} \sum_{\vec{x}} U_0 ^\dagger (T-1,\vec{x}) P_+ S(T-1,\vec{x})
```

is computed by the function [`bndtobnd`](@ref)

```@docs
bndtobnd
```

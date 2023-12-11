# Spinors

The module Spinors defines the necessary functions for the structure `Spinor{NS,G}`,
which is a NS-tuple with values in G.

The functions `norm`, `norm2`, `dot`, `*`, `/`, `/`, `+`, `-`, `imm` and `mimm`, 
if defined for G, are extended to Spinor{NS,G} for general NS.

For the 4d case where NS = 4 there are some specific functions to implement different
operations with the gamma matrices. The convention for these matrices is


```math
\gamma _4 = \left( 
    \begin{array}{cccc}
        0 & 0 & -1 & 0\\
        0 & 0 & 0 & -1\\
        -1 & 0 & 0 & 0\\
        0 & -1 & 0 & 0\\
    \end{array}
    \right)
    \quad 
    \gamma_1 = \left( 
    \begin{array}{cccc}
        0 & 0 & 0 & -i\\
        0 & 0 & -i & 0\\
        0 & i & 0 & 0\\
        i & 0 & 0 & 0\\
    \end{array}
    \right)
```
```math 
    \gamma _2 = \left( 
    \begin{array}{cccc}
        0 & 0 & 0 & -1\\
        0 & 0 & 1 & 0\\
        0 & 1 & 0 & 0\\
        -1 & 0 & 0 & 0\\
    \end{array}
    \right)
    \quad 
    \gamma_3 = \left( 
    \begin{array}{cccc}
        0 & 0 & -i & 0\\
        0 & 0 & 0 & i\\
        i & 0 & 0 & 0\\
        0 & -i & 0 & 0\\
    \end{array}
    \right)
```


The function [`dmul`](@ref) implements the multiplication over the $$\gamma$$ matrices

```@docs
dmul
```

The function [`pmul`](@ref) implements the $$ (1 \pm \gamma_N) $$ proyectors. The functions 
[`gpmul`](@ref) and [`gdagpmul`](@ref) do the same and then multiply each element by `g`and 
g^-1 repectively.

```@docs
pmul
gpmul
gdagpmul
```

## Some examples

Here we just display some examples for these functions. We display it with `ComplexF64`
instead of `SU3fund` or `SU2fund` for simplicity.


```@setup exs
import Pkg # hide
Pkg.activate("/home/alberto/code/julia/LatticeGPU/") # hide
using LatticeGPU # hide
```
```@repl exs
spin = Spinor{4,Complex{Float64}}((1.0,im*0.5,2.3,0.0))
println(spin)
println(dmul(Gamma{4},spin))
println(pmul(Pgamma{2,-1},spin))

```

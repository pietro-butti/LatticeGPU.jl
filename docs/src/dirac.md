
# Dirac operator

The module `Dirac` has the necessary stuctures and function 
to simulate non-dynamical 4-dimensional Wilson fermions.

There are two main data structures in this module, the structure [`DiracParam`](@ref)

```@docs
DiracParam
```

and the workspace [`DiracWorkspace`](@ref)

```@docs
DiracWorkspace
```

The workspace stores four fermion fields, namely `.sr`, `.sp`, `.sAp` and `.st`, used
for different purposes. If the representation is either `SU2fund` of `SU3fund`, an extra
field with values in `U2alg`/`U3alg` is created to store the clover, used for the improvent.

## Functions

The functions [`Dw!`](@ref), [`g5Dw!`](@ref) and [`DwdagDw!`](@ref) are all related to the 
Wilson-Dirac operator.

The action of the Dirac operator `Dw!` is the following:

```math
D_w\psi (\vec{x} = x_1,x_2,x_3,x_4) = (4 + m_0)\psi(\vec{x}) - 
```
```math
    - \frac{1}{2}\sum_{\mu = 1}^4 \theta (\mu) (1-\gamma_\mu) U_\mu(\vec{x}) \psi(\vec{x} + \hat{\mu}) + \theta^* (\mu) (1 + \gamma_\mu) U^{-1}_\mu(\vec{x} - \hat{\mu}) \psi(\vec{x} - \hat{\mu})
```

where $$m_0$$ and $$\theta$$ are respectively the values `.m0` and `.th` of [`DiracParam`](@ref).
Note that $$|\theta(\mu)|=1$$ is not built into the code, so it should be imposed explicitly. 

Additionally, if |`dpar.csw`| > 1.0E-10, the clover term is assumed to be stored in `ymws.csw`, which
can be done via the [`Csw!`](@ref) function. In this case we have the Sheikholeslami–Wohlert (SW) term
in `Dw!`:

```math
    \delta D_w^{sw} = \frac{i}{2}c_{sw} \sum_{\pi = 1}^6 F^{cl}_\pi \sigma_\pi \psi(\vec{x})
```
where the $$\sigma$$ matrices are those described in the `Spinors` module and the index $$\pi$$ runs
as specified in `lp.plidx`.

If the boudary conditions, defined in `lp`, are either `BC_SF_ORBI,D` or `BC_SF_AFWB`, the 
improvement term 

```math
    \delta D_w^{SF} = (c_t -1) (\delta_{x_4,a} \psi(\vec{x}) + \delta_{x_4,T-a} \psi(\vec{x}))
```
is added. Since the time-slice $$t=T$$ is not stored, this accounts to modifying the second
and last time-slice. 

Note that the Dirac operator for SF boundary conditions assumes that the value of the field 
in the first time-slice is zero. To enforce this, we have the function

```@docs
SF_bndfix!
```

Note that this is not enforced in the Dirac operators, so if the field `so` does not satisfy SF
boundary conditions, it will not (in general) satisfy them after applying [`Dw!`](@ref) 
or [`g5Dw!`](@ref). This function is called for the function [`DwdagDw!`](@ref), so in this case
`so` will always be a proper SF field after calling this function.

The function [`Csw!`](@ref) is used to store the clover in `dws.csw`. It is computed 
according to the expression

```math
F_{\mu,\nu} = \frac{1}{8} (Q_{\mu \nu} - Q_{\nu \mu})
```

where 
```math
Q_{\mu\nu} = U_\mu(\vec{x})U_{\nu}(x+\mu)U_{\mu}^{-1}(\vec{x}+\nu)U_{\nu}(\vec{x}) +
U_{\nu}^{-1}(\vec{x}-\nu) U_\mu (\vec{x}-\nu) U_{\nu}(\vec{x} +\mu - \nu) U^{-1}_{\mu}(\vec{x}) +
```
```math
+ U^{-1}_{\mu}(x-\mu)U_\nu^{-1}(\vec{x} - \mu - \nu)U_\mu(\vec{x} - \mu - \nu)U_\nu^{-1}(x-\nu) +
```
```math
+U_{\nu}(\vec{x})U_{\mu}^{-1}(\vec{x} + \nu - \mu)U^{-1}_{\nu}(\vec{x} - \mu)U_\mu(\vec{x}-\mu)

```

The correspondence between the tensor field and the GPU-Array is the following:
```math
F[b,1,r] \to F_{41}(b,r) ,\quad F[b,2,r] \to F_{42}(b,r) ,\quad F[b,3,r] \to F_{43}(b,r) 
```
```math
F[b,4,r] \to F_{31}(b,r) ,\quad F[b,5,r] \to F_{32}(b,r) ,\quad F[b,6,r] \to F_{21}(b,r) 
```
where $$(b,r)$$ labels the lattice points as explained in the module `Space`

The function [`pfrandomize!`](@ref), userfull for stochastic sources, is also present. It
randomizes a  fermion field either in all the space or in a specifit time-slice.

The generic interface of these functions reads

```@docs
Dw!
g5Dw!
DwdagDw!
Csw!
pfrandomize!
```

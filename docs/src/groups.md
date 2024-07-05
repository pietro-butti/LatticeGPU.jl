
# Groups and Algebras

The module `Groups` contains generic data types to deal with group and
algebra elements. Group elements $$g\in SU(N)$$ are represented in
some compact notation. For the case $$N=2$$ we use two complex numbers
(Caley-Dickson representation, i.e. $$g=(z_1,z_2)$$ with
$$|z_1|^2 + |z_2|^2=1$$). For the case $$N=3$$ we only store two
rows of the $$N\times N$$ unitary matrix.

Group operations translate straightforwardly in matrix operations,
except for $$SU(2)$$, where we use
```math
\forall g=(z_1, z_2)\in SU(2)\, \Longrightarrow\, g^{-1} =
(\overline z_1, -z_2)
```
and 
```math
\forall g=(z_1, z_2), g'=(w_1,w_2)\in SU(2)\, \Longrightarrow \, g g'= (z_1w_1 - \overline w_2 z_2, w_2z_1 + z_2\overline w_1)\,.
```

Group elements can be "casted" into arrays. The abstract type
[`GMatrix`](@ref) represent generic $$ N\\times N$$ complex arrays.


Algebra elements $$X \in \mathfrak{su}(N)$$ are traceless
anti-hermitian matrices represented trough $$N^2-1$$ real numbers
$$X^a$$. We have
```math
  X = \sum_{a=1}^{N^2-1} X^a T^a
```
where $$T^a$$ are a basis of the $$\mathfrak{su}(N)$$ algebra with
the conventions
```math
  (T^a)^+ = -T^a \,\qquad
  {\rm Tr}(T^aT^b) = -\frac{1}{2}\delta_{ab}\,.
```

If we denote by $$\{\mathbf{e}_i\}$$ the usual ortonormal basis of
$$\mathbb R^N$$, the $$N^2-1$$ matrices $$T^a$$ can always be written in the
following way 
1. symmetric matrices ($$N(N-1)/2$$) 
   ```math
   (T^a)_{ij} = \frac{\imath}{2}(\mathbf{e}_i\otimes\mathbf{e}_j +
   \mathbf{e}_j\otimes \mathbf{e}_i)\quad (1\le i < j \le N)
   ```
1. anti-symmetric matrices ($$N(N-1)/2$$) 
   ```math
   (T^a)_{ij} = \frac{1}{2}(\mathbf{e}_i\otimes\mathbf{e}_j -
   \mathbf{e}_j\otimes \mathbf{e}_i)\quad (1\le i < j \le N)
   ```
1. diagonal matrices ($$N-1$$). With $$l=1,...,N-1$$ and $$a=l+N(N-1)$$
   ```math
   (T^a)_{ij} = \frac{\imath}{2}\sqrt{\frac{2}{l(l+1)}}
   \left(\sum_{j=1}^l\mathbf{e}_j\otimes\mathbf{e}_j -
     l\mathbf{e}_{l+1}\otimes \mathbf{e}_{l+1}\right)
   \quad (l=1,\dots,N-1;\, a=l+N(N-1))
   ```

For example in the case of $$\mathfrak{su}(2)$$, and denoting the Pauli
matrices by $$\sigma^a$$ we have
```math
  T^a = \frac{\imath}{2}\sigma^a \quad(a=1,2,3)\,,
```
while for $$\mathfrak{su}(3)$$ we have $$T^a=\imath \lambda^a/2$$ (with
$$\lambda^a$$ the Gell-Mann matrices) up to a permutation.


## Some examples

```@setup exs
import Pkg # hide
Pkg.activate("/home/alberto/code/julia/LatticeGPU/") # hide
using LatticeGPU # hide
```

Here we just show some example codes playing with groups and
elements. The objective is to get an idea on how group operations 
We can generate some random group elements.
```@repl exs
# Generate random groups elements, 
# check they are actually from the group
g = rand(SU2{Float64})
println("Are we in a group?: ", isgroup(g))
g = rand(SU3{Float64})
println("Are we in a group?: ", isgroup(g))
```

We can also do some simple operations 
```@repl exs
X = rand(SU3alg{Float64})
g1 = exp(X);
g2 = exp(X, -2.0); # e^{-2X}
g = g1*g1*g2;
println("Is this one? ", dev_one(g))
```

## Types 

### Groups

The generic interface reads

```@docs
Group
```

Concrete supported implementations are
```@docs
SU2
SU3
```

### Algebras

The generic interface reads
```@docs
Algebra
```

With the following concrete implementations
```@docs
SU2alg
SU3alg
```

### GMatrix

The generic interface reads
```@docs
GMatrix
```

With the following concrete implementations
```@docs
M2x2
M3x3
```


## Generic `Group` methods

```@docs
inverse
dag
tr
dev_one
unitarize
isgroup
projalg
```

## Generic `Algebra` methods

```@docs
dot
norm
norm2
normalize
exp
expm
alg2mat
```



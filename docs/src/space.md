
# Lattice structure

D-dimensional lattice points are labeled by two ordered integer
numbers: the point-in-block index ($$b$$ in the figure below) and the
block index ($$r$$ in the figure below). The routines [`up`](@ref) and
[`dw`](@ref) allow you to displace to the neighboring points of the
lattice. 
![D dimensional lattice points are labeled by its
index (a `Tuple` of integers). Given a point (by its index), there are
routines to jump to neighboring points.](blocks.png)

Directions are numbered fro 1 to D, and then Euclidean time is always
assumed to be the last coordinate. Planes are also numbered from 1 to
$$N(N-1)/2$$. The structure [`SpaceParm`](@ref)
contains the information of the 
lattice structure and has to be passed as argument to most routines. 

```@docs
SpaceParm
up
dw
updw
point_coord
point_time
point_index
point_color
```

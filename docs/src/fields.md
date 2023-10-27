
# Lattice fields

The module `Fields` include simple routines to define a few typical
fields. Fields are simple `CuArray` types with special memory
layout. A field always has an associated elemental type (i.e. for 
gauge fields `SU3`, for scalar fields `Float64`). We have:
- scalar fields: One elemental type in each spacetime point.
- vector field: One elemental type at each spacetime point and
  direction.
- `N` scalar fields: `N` elemental types at each spacetime point.

For all these fields the spacetime point are ordered in memory
according to the point-in-block and block indices (see
[`SpaceParm`](@ref)). An execption is the [`scalar_field_point`](@ref)
fields. 

## Initialization

```@docs
scalar_field
vector_field
nscalar_field
scalar_field_point
```

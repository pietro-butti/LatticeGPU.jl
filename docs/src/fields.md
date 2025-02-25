
# Lattice fields

The module `Fields` includes simple routines to define a few typical
fields. Fields are simple `CuArray` types with special memory
layout. A field always has an associated elemental type (i.e. for 
gauge fields `SU3`, for scalar fields `Float64`). We have:
- Scalar fields: One elemental type in each spacetime point.
- Vector field: One elemental type at each spacetime point and
  direction.
- `N` scalar fields: `N` elemental types at each spacetime point.
- Tensor fields: One elemental type at each spacetime point and
  plane. They are to be thought of as symmetric tensors.

Fields can have **natural indexing**, where the memory layout follows
the point-in-block and block indices (see
[`SpaceParm`](@ref)). Fields can also have **lexicographic indexing**,
where points are labelled by a D-dimensional index (see [`scalar_field_point`](@ref)).


## Initialization

```@docs
scalar_field
vector_field
tensor_field
nscalar_field
scalar_field_point
```

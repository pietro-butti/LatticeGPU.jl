###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Fields.jl
### created: Wed Oct  6 17:37:03 2021
###                               

module Fields

using CUDA
using ..Space

"""
        vector_field(::Type{T}, lp::SpaceParm)

Returns a vector field of elemental type `T`. 
"""
vector_field(::Type{T}, lp::SpaceParm)     where {T} = CuArray{T, 3}(undef, lp.bsz, lp.ndim, lp.rsz)

"""
        scalar_field(::Type{T}, lp::SpaceParm)

Returns a scalar field of elemental type `T`.
"""
scalar_field(::Type{T}, lp::SpaceParm)     where {T} = CuArray{T, 2}(undef, lp.bsz, lp.rsz)

"""
        nscalar_field(::Type{T}, n::Integer, lp::SpaceParm)

Returns `n` scalar fields of elemental type `T`
"""
nscalar_field(::Type{T}, n, lp::SpaceParm) where {T} = CuArray{T, 3}(undef, lp.bsz, n, lp.rsz)


"""
        scalar_field(::Type{T}, lp::SpaceParm)

Returns a scalar field of elemental type `T`, with lexicografic memory layout.
"""
scalar_field_point(::Type{T}, lp::SpaceParm{N,M,D}) where {T,N,M,D} = CuArray{T, N}(undef, lp.iL...)

"""
        tensor_field(::Type{T}, lp::SpaceParm)

Returns a tensor field of elemental type `T`.
"""
tensor_field(::Type{T}, lp::SpaceParm)     where {T} = CuArray{T, 3}(undef, lp.bsz, lp.npls, lp.rsz)


export vector_field, scalar_field, nscalar_field, scalar_field_point, tensor_field

end


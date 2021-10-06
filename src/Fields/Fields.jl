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


vector_field(::Type{T}, lp::SpaceParm)     where {T} = CuArray{T, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
scalar_field(::Type{T}, lp::SpaceParm)     where {T} = CuArray{T, 3}(undef, lp.bsz, lp.rsz)
nscalar_field(::Type{T}, n, lp::SpaceParm) where {T} = CuArray{T, 3}(undef, lp.bsz, n, lp.rsz)

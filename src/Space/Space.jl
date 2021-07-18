###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Space.jl
### created: Mon Jul 12 16:44:35 2021
###                               


module Space

struct SpaceParm{N,M}
    ndim::Int64
    iL::NTuple{N,Int64}
    npls::Int64
    plidx::NTuple{M, Tuple{Int64, Int64}}

    function SpaceParm{N}(x) where {N}
        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Tuple of incorrect length for dimension $N"))

        pls = Vector{Tuple{Int64, Int64}}()
        for i in 1:N
            for j in i+1:N
                push!(pls, (i,j))
            end
        end
        return new{N,M}(N, x, M, tuple(pls...))
    end
end
export SpaceParm

struct KernelParm
    threads::Tuple{Int64,Int64,Int64}
    blocks::Tuple{Int64,Int64,Int64}
end
export KernelParm

@inline shift(p::CartesianIndex{4}, sh::CartesianIndex{4}, lp::SpaceParm{4}) = CartesianIndex(mod1(p[1]+sh[1], lp.iL[1]),
                                                                                          mod1(p[2]+sh[2], lp.iL[2]),
                                                                                          mod1(p[3]+sh[3], lp.iL[3]),
                                                                                          mod1(p[4]+sh[4], lp.iL[4]))
@inline shift(p::CartesianIndex{3}, sh::CartesianIndex{3}, lp::SpaceParm{3}) = CartesianIndex(mod1(p[1]+sh[1], lp.iL[1]),
                                                                                          mod1(p[2]+sh[2], lp.iL[2]),
                                                                                          mod1(p[3]+sh[3], lp.iL[3]))
@inline shift(p::CartesianIndex{2}, sh::CartesianIndex{2}, lp::SpaceParm{2}) = CartesianIndex(mod1(p[1]+sh[1], lp.iL[1]),
                                                                                          mod1(p[2]+sh[2], lp.iL[2]))
@inline function dw(p::CartesianIndex{4}, id, lp::SpaceParm{4})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]-1, lp.iL[1]), p[2], p[3], p[4])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]-1, lp.iL[2]), p[3], p[4])
    elseif (id == 3)
        s = CartesianIndex(p[1], p[2], mod1(p[3]-1, lp.iL[3]), p[4])
    elseif (id == 4)
        s = CartesianIndex(p[1], p[2], p[3], mod1(p[4]-1, lp.iL[4]))
    end
    
    return s
end

@inline function dw(p::CartesianIndex{3}, id, lp::SpaceParm{3})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]-1, lp.iL[1]), p[2], p[3])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]-1, lp.iL[2]), p[3])
    elseif (id == 3)
        s = CartesianIndex(p[1], p[2], mod1(p[3]-1, lp.iL[3]))
    end
    
    return s
end

@inline function dw(p::CartesianIndex{2}, id, lp::SpaceParm{2})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]-1, lp.iL[1]), p[2])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]-1, lp.iL[2]))
    end
    
    return s
end

@inline function up(p::CartesianIndex{4}, id::Int64, lp::SpaceParm{4})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]+1, lp.iL[1]), p[2], p[3], p[4])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]+1, lp.iL[2]), p[3], p[4])
    elseif (id == 3)
        s = CartesianIndex(p[1], p[2], mod1(p[3]+1, lp.iL[3]), p[4])
    elseif (id == 4)
        s = CartesianIndex(p[1], p[2], p[3], mod1(p[4]+1, lp.iL[4]))
    end

    return s
end

@inline function up(p::CartesianIndex{3}, id::Int64, lp::SpaceParm{3})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]+1, lp.iL[1]), p[2], p[3])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]+1, lp.iL[2]), p[3])
    elseif (id == 3)
        s = CartesianIndex(p[1], p[2], mod1(p[3]+1, lp.iL[3]))
    end

    return s
end

@inline function up(p::CartesianIndex{2}, id::Int64, lp::SpaceParm{2})

    if (id == 1)
        s = CartesianIndex(mod1(p[1]+1, lp.iL[1]), p[2])
    elseif (id == 2)
        s = CartesianIndex(p[1], mod1(p[2]+1, lp.iL[2]))
    end

    return s
end

@inline map2latt(th::NTuple{3,Int64},bl::NTuple{3,Int64}) = CartesianIndex(th[1],bl[1],bl[2],bl[3])

export map2latt, up, dw, shift

end

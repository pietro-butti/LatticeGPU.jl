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

import Base.show

struct SpaceParm{N,M}
    ndim::Int64
    iL::NTuple{N,Int64}
    npls::Int64
    plidx::NTuple{M,Tuple{Int64, Int64}}

    blk::NTuple{N,Int64}
    blkS::NTuple{N,Int64}
    rbk::NTuple{N,Int64}
    rbkS::NTuple{N,Int64}

    bsz::Int64
    rsz::Int64
    
    function SpaceParm{N}(x, y) where {N}
        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Lattice size incorrect length for dimension $N"))
        N == length(y) || throw(ArgumentError("Block   size incorrect length for dimension $N"))

        pls = Vector{Tuple{Int64, Int64}}()
        for i in 1:N
            for j in i+1:N
                push!(pls, (i,j))
            end
        end

        r  = div.(x, y)
        rS = ones(N)
        yS = ones(N)
        for i in 2:N
            for j in 1:i-1
                rS[i] = rS[i]*r[j]
                yS[i] = yS[i]*y[j]
            end
        end

        return new{N,M}(N, x, M, tuple(pls...), y,
                        tuple(yS...), tuple(r...), tuple(rS...), prod(y), prod(r))
    end
end
export SpaceParm
function Base.show(io::IO, lp::SpaceParm)
    println(io, "Lattice dimensions: ", lp.ndim)

    print(io, "Lattice size:       ")
    for i in 1:lp.ndim-1
        print(io, lp.iL[i], " x ")
    end
    println(io,lp.iL[end])
    
    print(io, "Thread block size:  ")
    for i in 1:lp.ndim-1
        print(io, lp.blk[i], " x ")
    end
    println(io,lp.blk[end], "     [", lp.bsz,
            "] (Number of blocks: [", lp.rsz,"])")
    return
end
    

@inline function up(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == lp.blk[id]-1)
        b = p[1] - (lp.blk[id]-1)*lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == lp.rbk[id]-1)
            r = p[2] - (lp.rbk[id]-1)*lp.rbkS[id]
        else
            r = p[2] + lp.rbkS[id]
        end
    else
        b = p[1] + lp.blkS[id]
        r = p[2]
    end

        
    return b, r
end

@inline function dw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == 0)
        b = p[1] + (lp.blk[id]-1)*lp.blkS[id]
        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == 0)
            r = p[2] + (lp.rbk[id]-1)*lp.rbkS[id]
        else
            r = p[2] - lp.rbkS[id]
        end
    else
        b = p[1] - lp.blkS[id]
        r = p[2]
    end

        
    return b, r
end

@inline function updw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == lp.blk[id]-1)
        bu = p[1] - (lp.blk[id]-1)*lp.blkS[id]
        bd = p[1] - lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == lp.rbk[id]-1)
            ru = p[2] - (lp.rbk[id]-1)*lp.rbkS[id]
        else
            ru = p[2] + lp.rbkS[id]
        end
        rd = p[2]
    elseif (ic == 0)
        bu = p[1] + lp.blkS[id]
        bd = p[1] + (lp.blk[id]-1)*lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == 0)
            rd = p[2] + (lp.rbk[id]-1)*lp.rbkS[id]
        else
            rd = p[2] - lp.rbkS[id]
        end
        ru = p[2]
    else
        bu = p[1] + lp.blkS[id]
        bd = p[1] - lp.blkS[id]
        rd = p[2]
        ru = p[2]
    end

        
    return bu, ru, bd, rd
end

@inline function global_point(p::NTuple{2,Int64}, lp::SpaceParm)

    @inline cntb(nb, id::Int64, lp::SpaceParm) = mod(div(nb-1,lp.blkS[id]),lp.blk[id])
    @inline cntr(nr, id::Int64, lp::SpaceParm) = mod(div(nr-1,lp.rbkS[id]),lp.rbk[id])

    @inline cnt(nb, nr, id::Int64, lp::SpaceParm)  = 1 + cntb(nb,id,lp) + cntr(nr,id,lp)*lp.blk[id]
    
    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return pt
end

@inline function point_idx(pt::NTuple{4, Int64}, lp::SpaceParm)

    
end
    
export up, dw, updw, global_point, point_idx

end

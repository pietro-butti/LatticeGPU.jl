###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMfields.jl
### created: Thu Jul 15 15:16:47 2021
###                               

un(t) = t <: Union{Group, Complex}

function field(::Type{T}, lp::SpaceParm) where {T <: Union{Group, Algebra}}

    sz = lp.bsz, lp.ndim, lp.rsz
    if (T == SU2)
#        As = StructArray{SU2}(undef, sz, unwrap=un)
        return CuArray{SU2, 3}(undef, sz)
    elseif (T == SU2alg)
#        As = StructArray{SU2alg}(undef, sz, unwrap=un)
        return CuArray{SU2alg, 3}(undef, sz)
    elseif (T == SU3)
#        As = StructArray{SU3}(undef, sz, unwrap=un)
        return CuArray{SU3, 3}(undef, sz)
#        As = Array{SU3, lp.ndim+1}(undef, sz)
    elseif (T == SU3alg)
        #        As = StructArray{SU3alg}(undef, sz, unwrap=un)
        return CuArray{SU3alg, 3}(undef, sz)
#        As = Array{SU3alg, lp.ndim+1}(undef, sz)
    end
        
    return replace_storage(CuArray, As)

end

function krnl_SU3_one!(G, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    for id in 1:lp.ndim
        G[b,id,r] = SU3(1.0,0.0,0.0,0.0,1.0,0.0)
    end
    return nothing
end

function krnl_SU2_one!(G, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    for id in 1:lp.ndim
        G[b,id,r] = SU2(1.0,0.0)
    end
    return nothing
end

function krnl_SU3alg_zero!(G, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    for id in 1:lp.ndim
        G[b,id,r] = SU3alg(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    end
    return nothing
end

function krnl_SU2alg_zero!(G, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    for id in 1:lp.ndim
        G[b,id,r] = SU2alg(0.0,0.0,0.0)
    end
    return nothing
end

function krnl_SU3alg_assign!(G, M, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    
    for id in 1:lp.ndim
        G[b,id,r] = SU3alg(M[b,id,r,1], M[b,id,r,2], M[b,id,r,3], M[b,id,r,4],
                           M[b,id,r,5], M[b,id,r,6], M[b,id,r,7], M[b,id,r,8])
    end
    return nothing
end

function krnl_SU2alg_assign!(G, M, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    
    for id in 1:lp.ndim
        G[b,id,r] = SU2alg(M[b,id,r,1], M[b,id,r,2], M[b,id,r,3])
    end
    return nothing
end

function randomn!(X, lp)

    if (eltype(X) == SU2alg)
#        randn!(CURAND.default_rng(), LazyRows(X).t1)
#        randn!(CURAND.default_rng(), LazyRows(X).t2)
#       randn!(CURAND.default_rng(), LazyRows(X).t3)
        M = CuArray{Float64}(undef, lp.bsz, lp.ndim, lp.rsz, 3)
        randn!(CURAND.default_rng(), M)        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU2alg_assign!(X, M, lp)
        end
    elseif (eltype(X) == SU3alg)
#        randn!(CURAND.default_rng(), LazyRows(X).t1)
#        randn!(CURAND.default_rng(), LazyRows(X).t2)
#        randn!(CURAND.default_rng(), LazyRows(X).t3)
#        randn!(CURAND.default_rng(), LazyRows(X).t4)
#        randn!(CURAND.default_rng(), LazyRows(X).t5)
#        randn!(CURAND.default_rng(), LazyRows(X).t6)
#        randn!(CURAND.default_rng(), LazyRows(X).t7)
#        randn!(CURAND.default_rng(), LazyRows(X).t8)
        M = CuArray{Float64}(undef, lp.bsz, lp.ndim, lp.rsz, 8)
        randn!(CURAND.default_rng(), M)        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU3alg_assign!(X, M, lp)
        end
    end
    return nothing
end

function zero!(X, lp)

    if (eltype(X) == SU2alg)
#        fill!(LazyRows(X).t1, 0.0)
#        fill!(LazyRows(X).t2, 0.0)
#        fill!(LazyRows(X).t3, 0.0)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU2alg_zero!(X, lp)
        end
    end

    if (eltype(X) == SU3alg)
#        fill!(LazyRows(X).t1, 0.0)
#        fill!(LazyRows(X).t2, 0.0)
#        fill!(LazyRows(X).t3, 0.0)
#        fill!(LazyRows(X).t4, 0.0)
#        fill!(LazyRows(X).t5, 0.0)
#        fill!(LazyRows(X).t6, 0.0)
#        fill!(LazyRows(X).t7, 0.0)
#        fill!(LazyRows(X).t8, 0.0)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU3alg_zero!(X, lp)
        end
    end

    if (eltype(X) == SU2)
#        fill!(LazyRows(X).t1.re, 1.0)
#        fill!(LazyRows(X).t1.im, 0.0)
#        fill!(LazyRows(X).t2.re, 0.0)
#        fill!(LazyRows(X).t2.im, 0.0)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU2_one!(X, lp)
        end
    end

    if (eltype(X) == SU3)
#        fill!(LazyRows(X).u11.re, 1.0)
        #        fill!(LazyRows(X).u11.im, 0.0)
        #        fill!(LazyRows(X).u12.re, 0.0)
#        fill!(LazyRows(X).u12.im, 0.0)
#        fill!(LazyRows(X).u13.re, 0.0)
#        fill!(LazyRows(X).u13.im, 0.0)
#        fill!(LazyRows(X).u21.re, 0.0)
#        fill!(LazyRows(X).u21.im, 0.0)
#        fill!(LazyRows(X).u22.re, 1.0)
#        fill!(LazyRows(X).u22.im, 0.0)
#        fill!(LazyRows(X).u23.re, 0.0)
#        fill!(LazyRows(X).u23.im, 0.0)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_SU3_one!(X, lp)
        end
    end
    
    return nothing
end
        
function norm_field(X)

    return CUDA.mapreduce(norm2, +, X)
#    d = 0.0
    if (eltype(X) == SU2alg)
#        d = CUDA.mapreduce(x->x^2, +, LazyRows(X).t1) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t2) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t3)
    elseif (eltype(X) == SU3alg)
#        d = CUDA.mapreduce(x->x^2, +, LazyRows(X).t1) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t2) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t3) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t4) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t5) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t6) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t7) +
#            CUDA.mapreduce(x->x^2, +, LazyRows(X).t8)
        d = CUDA.mapreduce(norm2, +, X)
    end
    
    return d
end

    

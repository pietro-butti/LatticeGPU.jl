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


function field(::Type{T}, lp::SpaceParm) where {T <: Union{Group, Algebra}}

    sz = lp.iL..., lp.ndim
    if (T == SU2)
        As = StructArray{SU2}((ones(ComplexF64, sz), zeros(ComplexF64, sz)))
    elseif (T == SU2alg)
        As = StructArray{SU2alg}((zeros(Float64, sz),
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz)))
    elseif (T == SU3)
        As = StructArray{SU3}((ones(ComplexF64, sz), zeros(ComplexF64, sz), zeros(ComplexF64, sz), zeros(ComplexF64, sz), ones(ComplexF64, sz), zeros(ComplexF64, sz)))
#        As = Array{SU3, lp.ndim+1}(undef, sz)
#        CUDA.@sync begin
#            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_SU3_zero!(As, lp)
#        end
    elseif (T == SU3alg)
        As = StructArray{SU3alg}((zeros(Float64, sz),
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz), 
                                  zeros(Float64, sz)))

#        As = Array{SU3alg, lp.ndim+1}(undef, sz)
#        CUDA.@sync begin
#            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_SU3alg_zero!(As, lp)
#        end
    end
        
    return replace_storage(CuArray, As)
end

function randomn!(X)

    if (eltype(X) == SU2alg)
        randn!(CURAND.default_rng(), LazyRows(X).t1)
        randn!(CURAND.default_rng(), LazyRows(X).t2)
        randn!(CURAND.default_rng(), LazyRows(X).t3)
    elseif (eltype(X) == SU3alg)
        randn!(CURAND.default_rng(), LazyRows(X).t1)
        randn!(CURAND.default_rng(), LazyRows(X).t2)
        randn!(CURAND.default_rng(), LazyRows(X).t3)
        randn!(CURAND.default_rng(), LazyRows(X).t4)
        randn!(CURAND.default_rng(), LazyRows(X).t5)
        randn!(CURAND.default_rng(), LazyRows(X).t6)
        randn!(CURAND.default_rng(), LazyRows(X).t7)
        randn!(CURAND.default_rng(), LazyRows(X).t8)
    end
    return nothing
end

function zero!(X)

    if (eltype(X) == SU2alg)
        fill!(LazyRows(X).t1, 0.0)
        fill!(LazyRows(X).t2, 0.0)
        fill!(LazyRows(X).t3, 0.0)
    end

    if (eltype(X) == SU3alg)
        fill!(LazyRows(X).t1, 0.0)
        fill!(LazyRows(X).t2, 0.0)
        fill!(LazyRows(X).t3, 0.0)
        fill!(LazyRows(X).t4, 0.0)
        fill!(LazyRows(X).t5, 0.0)
        fill!(LazyRows(X).t6, 0.0)
        fill!(LazyRows(X).t7, 0.0)
        fill!(LazyRows(X).t8, 0.0)
#        CUDA.@sync begin
#            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_SU3alg_zero!(X, lp)
#        end
    end

    if (eltype(X) == SU2)
        fill!(LazyRows(X).t1, complex(1.0))
        fill!(LazyRows(X).t2, complex(0.0))
    end

    if (eltype(X) == SU3)
        fill!(LazyRows(X).u11, complex(1.0))
        fill!(LazyRows(X).u12, complex(0.0))
        fill!(LazyRows(X).u13, complex(0.0))
        fill!(LazyRows(X).u21, complex(0.0))
        fill!(LazyRows(X).u22, complex(1.0))
        fill!(LazyRows(X).u23, complex(0.0))
#        CUDA.@sync begin
#            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_SU3_zero!(X, lp)
#        end
    end
    
    return nothing
end
        
function norm2(X)

    d = 0.0
    if (eltype(X) == SU2alg)
        d = CUDA.mapreduce(x->x^2, +, LazyRows(X).t1) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t2) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t3)
    elseif (eltype(X) == SU3alg)
        d = CUDA.mapreduce(x->x^2, +, LazyRows(X).t1) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t2) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t3) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t4) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t5) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t6) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t7) +
            CUDA.mapreduce(x->x^2, +, LazyRows(X).t8)
#        d = CUDA.mapreduce(norm2, +, X)
    end
    
    return d
end

function krnl_SU3_zero!(G, lp::SpaceParm)

    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    for id in 1:lp.ndim
        G[X,id].u11 = complex(1.0)
        G[X,id].u12 = complex(0.0)
        G[X,id].u13 = complex(0.0)
        G[X,id].u21 = complex(0.0)
        G[X,id].u22 = complex(1.0)
        G[X,id].u23 = complex(0.0)
    end
    return nothing
end

function krnl_SU3alg_zero!(G, lp::SpaceParm)

    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    for id in 1:lp.ndim
        G[X,id].t1 = 0.0
        G[X,id].t2 = 0.0
        G[X,id].t3 = 0.0
        G[X,id].t4 = 0.0
        G[X,id].t5 = 0.0
        G[X,id].t6 = 0.0
        G[X,id].t7 = 0.0
        G[X,id].t8 = 0.0
    end
    return nothing
end

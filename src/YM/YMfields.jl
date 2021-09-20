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

function field(::Type{G}, ::Type{T}, lp::SpaceParm) where {G <: Union{Group, Algebra}, T <: AbstractFloat}

    sz = lp.bsz, lp.ndim, lp.rsz
    if (G == SU2)
#        As = StructArray{SU2}(undef, sz, unwrap=un)
        return CuArray{SU2{T}, 3}(undef, sz)
    elseif (G == SU2alg)
#        As = StructArray{SU2alg}(undef, sz, unwrap=un)
        return CuArray{SU2alg{T}, 3}(undef, sz)
    elseif (G == SU3)
#        As = StructArray{SU3}(undef, sz, unwrap=un)
        return CuArray{SU3{T}, 3}(undef, sz)
#        As = Array{SU3, lp.ndim+1}(undef, sz)
    elseif (G == SU3alg)
        #        As = StructArray{SU3alg}(undef, sz, unwrap=un)
        return CuArray{SU3alg{T}, 3}(undef, sz)
#        As = Array{SU3alg, lp.ndim+1}(undef, sz)
    end
    
    return replace_storage(CuArray, As)

end

function randomize!(f, lp::SpaceParm, ymws::YMworkspace) 
        
    if ymws.ALG == SU2alg
        m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,3,lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU2!(f,m,lp)
        end
        return nothing
    end

    if ymws.ALG == SU3alg
        m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,8,lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU3!(f,m,lp)
        end
        return nothing
    end

    return nothing
end

function krnl_assign_SU3!(frc, m, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
                             m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
                             m[b,id,7,r], m[b,id,8,r])
    end
    return nothing
end

function krnl_assign_SU2!(frc, m, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        frc[b,id,r] = SU2alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r])
    end
    return nothing
end

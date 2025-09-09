###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMupdate.jl
### created: Mon Sep  1 08:59:22 2025
###                               

struct EvenOdd{T}
    state::Bool
    e::T
    o::T
end

function evenodd(eo::EvenOdd, lp::SpaceParm)

    if eo.state
        return eo
    end

    return evenodd(lp)
end

function evenodd(lp::SpaceParm)

    if all(lp.iL .% 2 .== 0)
        e = Array{Int64,1}(undef, div(lp.bsz,2))
        o = Array{Int64,1}(undef, div(lp.bsz,2))

        ie = 1
        io = 1
        for b in 1:lp.bsz
            clr = point_color((b, 1), lp)
            if clr % 2 == 0
                e[ie] = b
                ie = ie + 1
            else
                o[io] = b
                io = io + 1
            end
        end
        return EvenOdd(true, CuArray(e), CuArray(o))
    else
        return EvenOdd(false,nothing, nothing)
    end

end

function updt_or_wilson!(U, gp::GaugeParm, eo::EvenOdd, lp::SpaceParm{N,M,B,D}) where {N,M,B,D}

    if eo.state
        @timeit "OR update (Wilson action [eo])" begin
            ztw = ztwist(gp, lp)
            for id in 1:N
                CUDA.@sync begin
                    CUDA.@cuda threads=div(lp.bsz,2) blocks=lp.rsz krnl_or_wilson!(U, id, eo.e, ztw, gp, lp)
                end
            end
            
            for id in 1:N
                CUDA.@sync begin
                    CUDA.@cuda threads=div(lp.bsz,2) blocks=lp.rsz krnl_or_wilson!(U, id, eo.o, ztw, gp, lp)
                end
            end
        end
    else
        updt_or_wilson!(U, gp, lp)
    end
        
    return nothing
end

function updt_or_wilson!(U, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {N,M,B,D}

    @timeit "OR update (Wilson action)" begin
        ztw = ztwist(gp, lp)
        for id in 1:N
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_or_wilson!(U, id, 0, ztw, gp, lp)
            end
        end
        
        for id in 1:N
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_or_wilson!(U, id, 1, ztw, gp, lp)
            end
        end
    end
        
    return nothing
end


function krnl_or_wilson!(U::AbstractArray{SU3{T}}, id, icl::Int64, ztw, gp::GaugeParm, lp::SpaceParm) where {T}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    clr = point_color((b, r), lp)
    if clr % 2 == icl
        stp = sum_staples(U, id, (b, r), ztw, gp, lp)
        U[b,id,r] = do_or(U[b,id,r], stp)
    end

    return nothing
end

function krnl_or_wilson!(U::AbstractArray{SU3{T}}, id, eo::AbstractArray, ztw, gp::GaugeParm, lp::SpaceParm) where {T}

    b = eo[Int64(CUDA.threadIdx().x)]
    r = Int64(CUDA.blockIdx().x)
    stp = sum_staples(U, id, (b, r), ztw, gp, lp)
    U[b,id,r] = do_or(U[b,id,r], stp)

    return nothing
end

@inline function overrelax_cd(w::Tuple{T,T}) where {T}

    w2n = abs(w[2])^2
    dw = abs(w[1])^2 + w2n
    if dw == 0.0
        return tuple(complex(zero(eltype(w))), complex(zero(eltype(w))))
    end

    return tuple( (conj(w[1])*conj(w[1]) - w2n)/dw,  (-conj(w[1])*w[2] - w[1]*w[2])/dw )
end


@inline function do_or(U::SU3{T}, stp) where {T}

    M = do_or12(U, stp)
    M = do_or13(M, stp)
    U = do_or23(M, stp)
    U = unitarize(U)
    
    return U
end

@inline function do_or12(U::SU3{T}, stp) where {T} 
    
    w11 = U.u11*stp.u11 + U.u12*stp.u21 + U.u13*stp.u31
    w12 = U.u11*stp.u12 + U.u12*stp.u22 + U.u13*stp.u32
    w22 = U.u21*stp.u12 + U.u22*stp.u22 + U.u23*stp.u32
    w21 = U.u21*stp.u11 + U.u22*stp.u21 + U.u23*stp.u31
    w = tuple(complex(real(w11+w22)/2, imag(w11-w22)/2), complex(real(w12-w21)/2,imag(w12+w21)/2))
    h = overrelax_cd(w)

    u11 = h[1]*U.u11 + h[2]*U.u21
    u12 = h[1]*U.u12 + h[2]*U.u22
    u13 = h[1]*U.u13 + h[2]*U.u23

    u21 = -conj(h[2])*U.u11 + conj(h[1])*U.u21
    u22 = -conj(h[2])*U.u12 + conj(h[1])*U.u22
    u23 = -conj(h[2])*U.u13 + conj(h[1])*U.u23

    u31 = conj(U.u12*U.u23 - U.u13*U.u22)
    u32 = conj(U.u13*U.u21 - U.u11*U.u23)
    u33 = conj(U.u11*U.u22 - U.u12*U.u21)

    return M3x3{T}(u11, u12, u13, u21, u22, u23, u31, u32, u33)
end

@inline function do_or13(M::M3x3{T}, stp) where {T}

    w11 = M.u11*stp.u11 + M.u12*stp.u21 + M.u13*stp.u31
    w12 = M.u11*stp.u13 + M.u12*stp.u23 + M.u13*stp.u33
    w22 = M.u31*stp.u13 + M.u32*stp.u23 + M.u33*stp.u33
    w21 = M.u31*stp.u11 + M.u32*stp.u21 + M.u33*stp.u31
    w = tuple(complex(real(w11+w22)/2, imag(w11-w22)/2), complex(real(w12-w21)/2,imag(w12+w21)/2))
    h = overrelax_cd(w)

    u11 = h[1]*M.u11 + h[2]*M.u31
    u12 = h[1]*M.u12 + h[2]*M.u32
    u13 = h[1]*M.u13 + h[2]*M.u33

    u21 = M.u21
    u22 = M.u22
    u23 = M.u23

    u31 = -conj(h[2])*M.u11 + conj(h[1])*M.u31
    u32 = -conj(h[2])*M.u12 + conj(h[1])*M.u32
    u33 = -conj(h[2])*M.u13 + conj(h[1])*M.u33

    return M3x3{T}(u11, u12, u13, u21, u22, u23, u31, u32, u33)
end

@inline function do_or23(M::M3x3{T}, stp) where {T}
    
    w11 = M.u21*stp.u12 + M.u22*stp.u22 + M.u23*stp.u32
    w12 = M.u21*stp.u13 + M.u22*stp.u23 + M.u23*stp.u33
    w22 = M.u31*stp.u13 + M.u32*stp.u23 + M.u33*stp.u33
    w21 = M.u31*stp.u12 + M.u32*stp.u22 + M.u33*stp.u32
    w = tuple(complex(real(w11+w22)/2, imag(w11-w22)/2), complex(real(w12-w21)/2,imag(w12+w21)/2))
    h = overrelax_cd(w)

    u11 = M.u11
    u12 = M.u12
    u13 = M.u13

    u21 = h[1]*M.u21 + h[2]*M.u31
    u22 = h[1]*M.u22 + h[2]*M.u32
    u23 = h[1]*M.u23 + h[2]*M.u33

    return SU3{T}(u11, u12, u13, u21, u22, u23)
end

@inline function sum_staples(U::AbstractArray{SU3{T}},im,pt::NTuple{2,Int64}, ztw, gp::GaugeParm, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {T,N,M,D}

    I = point_coord(pt, lp)
    bum, rum = up(pt, im, lp)
    stp = zero(M3x3{T})
    for in in 1:N
        if (in != im)
            zf = one(ztw[begin])
            for i in 1:length(lp.plidx)
                if (lp.plidx[i] == (in, im)) || (lp.plidx[i] == (im, in))
                    if im > in
                        zf = ztw[i]
                    else
                        zf = conj(ztw[i])
                    end
                    break
                end
            end
            
            bun, run, bdn, rdn = updw(pt, in, lp)
            bd, rd = up((bdn, rdn), im, lp)
            m1 = convert(M3x3{T}, U[bum,in,rum]/U[bun,im,run]/U[pt[1],in,pt[2]])
            if (I[im]==1) && (I[in]==1)
                m1 = m1 * zf
            end
            m2 = convert(M3x3{T},dag(U[bdn,in,rdn]\U[bdn,im,rdn]*U[bd, in, rd]))
            Id = point_coord((bdn, rdn), lp)
            if (Id[im]==1) && (Id[in]==1)
                m2 = m2 * conj(zf)
            end
            
            stp = stp + m1 + m2
        end
    end
    
    return stp
end

@inline function sum_staples(U::AbstractArray{SU3{T}},im,pt::NTuple{2,Int64}, ztw, gp::GaugeParm, lp::SpaceParm{N,M,BC_SF_ORBI,D}) where {T,N,M,D}

    I = point_coord(pt, lp)
    bum, rum = up(pt, im, lp)
    stp = zero(M3x3{T})
    for in in 1:N
        if (in != im)
            bun, run, bdn, rdn = updw(pt, in, lp)
            bd, rd = up((bdn, rdn), im, lp)
           
            if (I[end] == lp.iL[end])
                if (in==N)
                    m1 = gp.cG[1]*convert(M3x3{T}, U[bum,in,rum]/gp.Ubnd[im]/U[pt[1],in,pt[2]])
                elseif (im==N)
                    m1 = gp.cG[1]*convert(M3x3{T}, gp.Ubnd[in]/U[bun,im,run]/U[pt[1],in,pt[2]])
                else
                    m1 = zero(M3x3{T})
                end
            else
                m1 = convert(M3x3{T}, U[bum,in,rum]/U[bun,im,run]/U[pt[1],in,pt[2]])
            end

            if I[end] == 1
                if (in==N) || (im==N)
                    m2 = gp.cG[2]*convert(M3x3{T},dag(U[bdn,in,rdn]\U[bdn,im,rdn]*U[bd, in, rd]))
                else
                    m2 = zero(M3x3{T})
                end
            else
                m2 = convert(M3x3{T},dag(U[bdn,in,rdn]\U[bdn,im,rdn]*U[bd, in, rd]))
            end
            stp = stp + m1 + m2

        end
    end
    
    return stp
end

@inline function sum_staples(U::AbstractArray{SU3{T}},im,pt::NTuple{2,Int64}, ztw, gp::GaugeParm, lp::SpaceParm{N,M,BC_OPEN,D}) where {T,N,M,D}

    I = point_coord(pt, lp)
    bum, rum = up(pt, im, lp)
    stp = zero(M3x3{T})
    for in in 1:N
        if (in != im)
            bun, run, bdn, rdn = updw(pt, in, lp)
            bd, rd = up((bdn, rdn), im, lp)
           
            if (I[end] == lp.iL[end])
                if (in==N)
                    m1 = gp.cG[1]*convert(M3x3{T}, U[bum,in,rum]/gp.Ubnd[im]/U[pt[1],in,pt[2]])
                else
                    m1 = zero(M3x3{T})
                end
            else
                m1 = convert(M3x3{T}, U[bum,in,rum]/U[bun,im,run]/U[pt[1],in,pt[2]])
            end

            if I[end] == 1
                if (in==N) || (im==N)
                    m2 = gp.cG[2]*convert(M3x3{T},dag(U[bdn,in,rdn]\U[bdn,im,rdn]*U[bd, in, rd]))
                else
                    m2 = zero(M3x3{T})
                end
            else
                m2 = convert(M3x3{T},dag(U[bdn,in,rdn]\U[bdn,im,rdn]*U[bd, in, rd]))
            end
            stp = stp + m1 + m2

        end
    end
    
    return stp
end



###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMsf.jl
### created: Tue Oct 26 14:50:55 2021
###                               

""" 
    sfcoupling(U, lp::SpaceParm{N,M,B,D}, gp::GaugeParm, ymws::YMworkspace) where {N,M,B,D}

Measures the Schrodinger Functional coupling `ds/d\eta` and `d^2S/d\eta d\nu`. 
"""
function sfcoupling(U, lp::SpaceParm{N,M,B,D}, gp::GaugeParm, ymws::YMworkspace) where {N,M,B,D}

    if lp.iL[end] < 4
        error("Array too small to store partial sums")
    end

    if !((B==BC_SF_AFWB) || (B==BC_SF_ORBI))
        error("SF coupling can only be measured with SF boundary conditions")
    end
    
    @timeit "SF coupling measurement" begin
        T = eltype(ymws.rm)
        tmp = zeros(T,lp.iL[end])
        fill!(ymws.rm, zero(T))
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sfcoupling!(ymws.rm, U, gp.Ubnd, lp)
        end

        tp = ntuple(i->i, N-1)
        V3 = prod(lp.iL[1:end-1])
        tmp .= reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])
        
        dsdeta = (gp.cG[1]*gp.beta/(2*gp.ng*V3))*(tmp[1] + tmp[end])
        ddnu   = (gp.cG[1]*gp.beta/(2*gp.ng*V3))*(tmp[2] + tmp[end-1])
    end
    
    return dsdeta, ddnu
end
    
function krnl_sfcoupling!(rm, U::AbstractArray{T}, Ubnd::T, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    I    = point_coord((b,r), lp)
    it   = I[N]

    SR3::eltype(rm)   = 1.73205080756887729352744634151
    SR3x2::eltype(rm) = 3.46410161513775458705489268302
    
    if (it == 1)
        but, rut = up((b,r), N, lp)
        IU = point_coord((but,rut), lp)
        for id in 1:N-1
            bu, ru = up((b,r), id, lp)

            X = projalg(U[b,id,r]*U[bu,N,ru]/(U[b,N,r]*U[but,id,rut]))
            rm[I]  += 3*X.t7 + SR3   * X.t8
            rm[IU] += 2*X.t7 - SR3x2 * X.t8
        end
    elseif (it == lp.iL[end])
        bdt, rdt = up((b,r), N, lp)
        ID = point_coord((bdt,rdt), lp)
        for id in 1:N-1
            bu, ru = up((b,r), id, lp)

            X = projalg(Ubnd/(U[b,id,r]*U[bu,N,ru])*U[b,N,r])
            rm[I]  -= 3*X.t7 + SR3   * X.t8
            rm[ID] += 2*X.t7 - SR3x2 * X.t8
        end
    end

    return nothing
end

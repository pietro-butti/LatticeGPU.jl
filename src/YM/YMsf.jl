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

const SR3   = 1.73205080756887729352744634151
const SR3x2 = 3.46410161513775458705489268302

function sfcoupling(U, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace)

    if lp.iL[end] < 4
        error("Array too small to store partial sums")
    end
    
    tmp = zeros(T,lp.iL[end])
    @timeit "SF coupling measurement" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sfcoupling!(ymws.rm, U, gp.Ubnd, lp)
        end
    end

    tp = ntuple(i->i, N-1)
    V3 = prod(lp.iL[1:end-1])
    tmp .= reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])

    dsdeta = (gp.beta/(2*gp.ng*V3))*(tmp[1] + tmp[end])
    ddnu   = (gp.beta/(2*gp.ng*V3))*(tmp[2] + tmp[end-1])
    
    return dsdeta, ddnu
end
    
function krnl_sfcoupling!(rm, U::AbstractArray{T}, Ubnd::T, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it   = point_time((b, r), lp)
    I    = point_coord((b,r), lp)
    
    rm[I] = zero(eltype(rm))
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

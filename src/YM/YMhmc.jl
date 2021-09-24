###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMhmc.jl
### created: Thu Jul 15 11:27:28 2021
###                               

function gauge_action(U, lp::SpaceParm, gp::GaugeParm{T}, ymws::YMworkspace{T}) where T <: AbstractFloat
    
    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq!(ymws.cm, U, lp)
    end
    S = gp.beta*( prod(lp.iL)*lp.npls -
                  CUDA.mapreduce(real, +, ymws.cm)/gp.ng )

    return S
end

function plaquette(U, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace)

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq!(ymws.cm, U, lp)
    end

    return CUDA.mapreduce(real, +, ymws.cm)/(prod(lp.iL)*lp.npls)
end
    
function hamiltonian(mom, U, lp, gp, ymws)
    K = CUDA.mapreduce(norm2, +, mom)/2 
    V = gauge_action(U, lp, gp, ymws)

    return K+V
end

function HMC!(U, eps, ns, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace; noacc=false)

    ymws.U1 .= U

    randomize!(ymws.mom, lp, ymws)
    hini = hamiltonian(ymws.mom, U, lp, gp, ymws)

    OMF4!(ymws.mom, U, eps, ns, lp, gp, ymws)
    
    dh   = hamiltonian(ymws.mom, U, lp, gp, ymws) - hini
    pacc = exp(-dh)

    acc = true
    if (noacc)
        return dh, acc
    end
    
    if (pacc < 1.0)
        r = rand()
        if (pacc < r) 
            U .= ymws.U1
            acc = false
        end
    end

    return dh, acc
end

function krnl_updt!(mom::AbstractArray{TF}, frc, eps1, U::AbstractArray{TU}, eps2, lp::SpaceParm{N,M,D}) where {TU,TF, N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(TU, D)
    Fsh = @cuStaticSharedMem(TF, D)
    
    @inbounds for id in 1:lp.ndim
        Ush[b] = U[b,id,r]
        Fsh[b] = frc[b,id,r]

        mom[b,id,r] = mom[b,id,r] + eps1 * Fsh[b]
        U[b,id,r] = expm(Ush[b], mom[b,id,r], eps2)
    end

    return nothing
end
                    
function OMF4!(mom, U, eps, ns, lp::SpaceParm, gp::GaugeParm{T}, ymws::YMworkspace{T}) where T <: AbstractFloat

    r1::T =  0.08398315262876693
    r2::T =  0.2539785108410595
    r3::T =  0.6822365335719091
    r4::T = -0.03230286765269967
    r5::T =  0.5-r1-r3
    r6::T =  1.0-2.0*(r2+r4)

    ee = eps*gp.beta/gp.ng
    force_wilson(ymws, U, lp)
    for i in 1:ns
        # STEP 1
#        CUDA.@sync begin
#            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_updt!(ymws.mom, ymws.frc1, r1*ee, U, eps*r2, lp)
#        end
        mom .= mom .+ (r1*ee) .* ymws.frc1
        U .= expm.(U, mom, eps*r2)
    
        # STEP 2
        force_wilson(ymws, U, lp)
#        CUDA.@sync begin
#            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_updt!(ymws.mom, ymws.frc1, r3*ee, U, eps*r4, lp)
#        end
        mom .= mom .+ (r3*ee) .* ymws.frc1
        U .= expm.(U, mom, eps*r4)

        # STEP 3
        force_wilson(ymws, U, lp)
#        CUDA.@sync begin
#            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_updt!(ymws.mom, ymws.frc1, r5*ee, U, eps*r6, lp)
#        end
        mom .= mom .+ (r5*ee) .* ymws.frc1
        U .= expm.(U, mom, eps*r6)

        # STEP 4
        force_wilson(ymws, U, lp)
#        CUDA.@sync begin
#            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_updt!(ymws.mom, ymws.frc1, r5*ee, U, eps*r4, lp)
#        end
        mom .= mom .+ (r5*ee) .* ymws.frc1
        U .= expm.(U, mom, eps*r4)

        # STEP 5
        force_wilson(ymws, U, lp)
#        CUDA.@sync begin
#            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_updt!(ymws.mom, ymws.frc1, r3*ee, U, eps*r2, lp)
#        end
        mom .= mom .+ (r3*ee) .* ymws.frc1
        U .= expm.(U, mom, eps*r2)

        # STEP 6
        force_wilson(ymws, U, lp)
        mom .= mom .+ (r1*ee) .* ymws.frc1
    end

    return nothing
end

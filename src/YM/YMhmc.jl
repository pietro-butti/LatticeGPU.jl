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

function gauge_action(U, lp::SpaceParm, gp::GaugeParm, kp::KernelParm, ymws::YMworkspace)
    
    CUDA.@sync begin
        CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_plaq!(ymws.cm, U, lp)
    end
    S = gp.beta*( prod(lp.iL)*lp.npls -
                  CUDA.mapreduce(real, +, real.(ymws.cm))/gp.ng )

    return S
end

function plaquette(U, lp::SpaceParm, gp::GaugeParm, kp::KernelParm, ymws::YMworkspace)

    CUDA.@sync begin
        CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_plaq!(ymws.cm, U, lp)
    end

    return CUDA.mapreduce(real, +, real.(ymws.cm))/(prod(lp.iL)*lp.npls)
end
    
hamiltonian(mom, U, lp, gp, kp, ymws) = norm2(mom)/2.0 +
    gauge_action(U, lp, gp, kp, ymws)

function HMC!(U, eps, ns, lp::SpaceParm, gp::GaugeParm, kp::KernelParm, ymws::YMworkspace; noacc=false)

    ymws.U1 .= U

    randomn!(ymws.mom)
    hini = hamiltonian(ymws.mom, U, lp, gp, kp, ymws)

    OMF4!(ymws.mom, U, eps, ns, lp, gp, kp, ymws)
    
    dh   = hamiltonian(ymws.mom, U, lp, gp, kp, ymws) - hini
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

function krnl_updt!(mom, frc1, frc2, eps1, U, eps2, lp::SpaceParm)

    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    for id in 1:lp.ndim
        mom[X,id]  = mom[X,id] + eps1 * (frc1[X,id]+frc2[X,id])
        U[X,id]    = expm(U[X,id], mom[X,id], eps2)
    end

    return nothing
end
                    
function OMF4!(mom, U, eps, ns, lp::SpaceParm, gp::GaugeParm, kp::KernelParm, ymws::YMworkspace)

    r1::Float64 =  0.08398315262876693
    r2::Float64 =  0.2539785108410595
    r3::Float64 =  0.6822365335719091
    r4::Float64 = -0.03230286765269967
    r5::Float64 =  0.5-r1-r3
    r6::Float64 =  1.0-2.0*(r2+r4)

    ee = eps*gp.beta/gp.ng
    force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
    for i in 1:ns
        # STEP 1
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r1*ee, U, eps*r2, lp)
        end

        # STEP 2
        force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r3*ee, U, eps*r4, lp)
        end

        # STEP 3
        force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r5*ee, U, eps*r6, lp)
        end

        # STEP 4
        force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r5*ee, U, eps*r4, lp)
        end

        # STEP 5
        force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r3*ee, U, eps*r2, lp)
        end

        # STEP 6
        force0_wilson!(ymws.frc1,ymws.frc2, U, lp, gp, kp)
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_updt!(ymws.mom, ymws.frc1,ymws.frc2, r1*ee, U, 0.0, lp)
        end
    end

    return nothing
end

###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMact.jl
### created: Mon Jul 12 18:31:19 2021
###                               

function krnl_plaq!(plx, U, ipl, lp::SpaceParm)

    id1, id2 = lp.plidx[ipl]
    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))
    Xu1 = up(X, id1, lp)
    Xu2 = up(X, id2, lp)

    plx[X] = tr(U[X, id1]*U[Xu1, id2] / (U[X, id2]*U[Xu2, id1]))

    return nothing
end

function krnl_plaq!(plx, U, lp::SpaceParm)
    
    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    plx[X] = complex(0.0)
    for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]
        Xu1 = up(X, id1, lp)
        Xu2 = up(X, id2, lp)
        
        plx[X] += tr(U[X, id1]*U[Xu1, id2] / (U[X, id2]*U[Xu2, id1]))
    end
    
    return nothing
end

function krnl_force_wilson_pln!(frc1, frc2, U, ipl, lp::SpaceParm, gp::GaugeParm)

    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    id1, id2 = lp.plidx[ipl]
    Xu1 = up(X, id1, lp)
    Xu2 = up(X, id2, lp)
    
    a = U[Xu1,id2]/U[Xu2,id1]
    b = U[X  ,id2]\U[X  ,id1]
    
    F1 = projalg(U[X,id1]*a/U[X,id2])
    F2 = projalg(a*b)
    F3 = projalg(b*a)
    
    frc1[X  ,id1] -= F1
    frc1[X  ,id2] += F1
    frc2[Xu1,id2] -= F2
    frc2[Xu2,id1] += F3

    return nothing
end

function force0_wilson!(frc1, frc2, U, lp::SpaceParm, gp::GaugeParm, kp::KernelParm)

    zero!(frc1)
    zero!(frc2)
    for ipl in 1:lp.npls
        CUDA.@sync begin
            CUDA.@cuda threads=kp.threads blocks=kp.blocks krnl_force_wilson_pln!(frc1,frc2,U,ipl,lp,gp)
        end
    end

    return nothing
end

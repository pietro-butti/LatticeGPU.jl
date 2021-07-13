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

    id1, id2 = lp.plidx(ipl)
    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))
    Xu1 = up(X, id1)
    Xu2 = up(X, id2)
    
    plx[X] = tr(U[X, id1]*U[Xu1, id2] / (U[X, id2]*U[Xu2, id1]))

    return nothing
end

function krnl_plaq!(plx, U, lp::SpaceParm)
    
    X = map2latt((CUDA.threadIdx().x,CUDA.threadIdx().y,CUDA.threadIdx().z),
                 (CUDA.blockIdx().x,CUDA.blockIdx().y,CUDA.blockIdx().z))

    plx[X] = 0.0
    for ipl in 1:lp.npls
        id1, id2 = lp.plidx(ipl)
        Xu1 = up(X, id1)
        Xu2 = up(X, id2)
        
        plx[X] += tr(U[X, id1]*U[Xu1, id2] / (U[X, id2]*U[Xu2, id1]))
    end
    plx[X] = plx[X]/lp.npls
    
    return nothing
end

###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMflow.jl
### created: Sat Sep 25 08:37:14 2021
###                               

"""
    function add_zth_term(ymws::YMworkspace, U, lp)

Assuming that the gauge improved (LW) force is in ymws.frc1, this routine
adds the "Zeuthen term" and returns the full zeuthen force in ymws.frc1
"""
function add_zth_term(ymws::YMworkspace, U, lp)

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_zth!(ymws.frc1,ymws.frc2,U,lp)
    end
    ymws.frc1 .= ymws.frc2 
    
    return nothing
end

function krnl_add_zth!(frc, frc2::AbstractArray{TA}, U::AbstractArray{TG}, lp::SpaceParm{N,M,D}) where {TA,TG,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(TG, D)
    Fsh = @cuStaticSharedMem(TA, D)
    
    @inbounds for id in 1:N
        Ush[b] = U[b,id,r]
        Fsh[b] = frc[b,id,r]
        sync_threads()
        
        bu, ru = up((b,r), id, lp)
        bd, rd = dw((b,r), id, lp)
        
        if ru == r
            X = Fsh[bu]
        else
            X = frc[bu,id,ru]
        end
        if rd == r
            Y  = Fsh[bd]
            Ud = Ush[bd]
        else
            Y  = frc[bd,id,rd]
            Ud = U[bd,id,rd]
        end
        
        frc2[b,id,r] = (5/6)*Fsh[b] + (1/6)*(projalg(Ud\Y*Ud) +
                                             projalg(Ush[b]*X/Ush[b]))
    end
    
    return nothing
end


function flw_euler(U, ns, eps, c0, lp::SpaceParm, ymws::YMworkspace; add_zth=false)
    
    for i in 1:ns
        force_gauge(ymws, U, c0, lp)
        if add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end
        U .= expm.(U, ymws.frc1, 2*eps)
    end
    
    return nothing
end

function flw_rk3(U, ns, eps, c0, lp::SpaceParm, ymws::YMworkspace; add_zth=false)
    
    for i in 1:ns
        e0 = eps/2
        force_gauge(ymws, U, c0, lp)
        if add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end
        ymws.mom .= ymws.frc1
        U .= expm.(U, ymws.mom, e0)
        
        e0 = -34*eps/36
        e1 = 16*eps/9
        force_gauge(ymws, U, c0, lp)
        if add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end
        ymws.mom .= e0.*ymws.mom .+ e1.*ymws.frc1
        U .= expm.(U, ymws.mom)
        
        e1 = 6*eps/4
        force_gauge(ymws, U, c0, lp)
        if add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end
        ymws.mom .= e1.*ymws.frc1 .- ymws.mom 
        U .= expm.(U, ymws.mom)
    end
                              
    return nothing
end



                              
wfl_euler(U, ns, eps, lp::SpaceParm, ymws::YMworkspace) = flw_euler(U, ns, eps, 1, lp, ymws)
zfl_euler(U, ns, eps, lp::SpaceParm, ymws::YMworkspace) = flw_euler(U, ns, eps, 5.0/3.0, lp, ymws, add_zth=true)
wfl_rk3(U, ns, eps, lp::SpaceParm, ymws::YMworkspace) = flw_rk3(U, ns, eps, 1, lp, ymws)
zfl_rk3(U, ns, eps, lp::SpaceParm, ymws::YMworkspace) = flw_rk3(U, ns, eps, 5.0/3.0, lp, ymws, add_zth=true)

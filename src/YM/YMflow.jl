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

function wfl_euler(U, ns, eps, lp::SpaceParm, ymws::YMworkspace)

    for i in 1:ns
        force_wilson(ymws, U, lp)
        U .= expm.(U, ymws.frc1, 2*eps)
    end
    
    return nothing
end

function wfl_rk3(U, ns, eps, lp::SpaceParm, ymws::YMworkspace)

    for i in 1:ns
        c0 = eps/2
        force_wilson(ymws, U, lp)
        ymws.mom .= ymws.frc1
        U .= expm.(U, ymws.mom, c0)

        c0 = -34*eps/36
        c1 = 16*eps/9
        force_wilson(ymws, U, lp)
        ymws.mom .= c0.*ymws.mom .+ c1.*ymws.frc1
        U .= expm.(U, ymws.mom)

        c1 = 6*eps/4
        force_wilson(ymws, U, lp)
        ymws.mom .= c1.*ymws.frc1 .- ymws.mom 
        U .= expm.(U, ymws.mom)
    end

    return nothing
end

function zfl_euler(U, ns, eps, lp::SpaceParm, ymws::YMworkspace)

    for i in 1:ns
        force_gauge(ymws, U, 5.0/3.0, lp)
        U .= expm.(U, ymws.frc1, 2*eps)
    end
    
    return nothing
end

function zfl_rk3(U, ns, eps, lp::SpaceParm, ymws::YMworkspace)

    for i in 1:ns
        c0 = eps/2
        force_gauge(ymws, U, 5.0/3.0, lp)
        ymws.mom .= ymws.frc1
        U .= expm.(U, ymws.mom, c0)

        c0 = -34*eps/36
        c1 = 16*eps/9
        force_gauge(ymws, U, 5.0/3.0, lp)
        ymws.mom .= c0.*ymws.mom .+ c1.*ymws.frc1
        U .= expm.(U, ymws.mom)

        c1 = 6*eps/4
        force_gauge(ymws, U, 5.0/3.0, lp)
        ymws.mom .= c1.*ymws.frc1 .- ymws.mom 
        U .= expm.(U, ymws.mom)
    end

    return nothing
end


###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    test_io.jl
### created: Thu Jun  2 18:20:42 2022
###                               

using LatticeGPU, Test

T = Float64
lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,0))
gp = GaugeParm{T}(SU3{T}, 6.1, 1.0)
ymws = YMworkspace(SU3, T, lp)

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom)

pl1 = plaquette(U, lp, gp, ymws)
cl1 = Eoft_clover(U, gp, lp, ymws)
fn = "foo.bdio"
rm(fn, force=true)
save_cnfg(fn, U, lp, gp; run="Dumnmy_run")

Ucp = read_cnfg(fn)
pl2 = plaquette(Ucp, lp, gp, ymws)
cl2 = Eoft_clover(Ucp, gp, lp, ymws)
rm(fn, force=true)

@testset "Testing configuration i/o" begin
    @test isapprox(pl1,pl2)
    @test isapprox(cl1,cl2)
end


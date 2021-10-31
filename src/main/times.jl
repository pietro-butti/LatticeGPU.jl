using CUDA, Logging, Random, TimerOutputs

CUDA.allowscalar(false)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

# Set lattice/block size
ntwist = (0,0,0,1,0,0)
lp = SpaceParm{4}((32,32,32,32), (4,4,4,4), BC_PERIODIC, ntwist)
println("Space  Parameters: ", lp)

# Seed RNG
println("Seeding CURAND...")
Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

# Set group and precision
GRP  = SU3
ALG  = SU3alg
PREC = Float64
println("Precision: ", PREC)

# MD integrator
int = omf4(PREC, 0.1, 10)
println(int)

println("Allocating YM workspace")
ymws = YMworkspace(GRP, PREC, lp)

# Main program
println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))

println("Time to take the configuration to memory: ")
@time Ucpu = Array(U)


# Set gauge parameters
# FIRST SET: Wilson action/flow
println("\n## WILSON ACTION/FLOW TIMES")
gp = GaugeParm{PREC}(GRP{PREC}, 6.0, 1.0, (0.5,0.5))
println("Gauge  Parameters: ", gp)


println("Initial Action: ")
@time S = gauge_action(U, lp, gp, ymws)


HMC!(U,int.eps,1,lp, gp, ymws, noacc=true)

pl = Vector{Float64}()
for i in 1:4
    @time dh, acc = HMC!(U,int,lp, gp, ymws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")

#    @time x, y = sfcoupling(U,lp,gp,ymws)
#    println("SF coupling: ", x, " ", y)
end

wfl_rk3(U, 1, 0.01, gp, lp, ymws)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("Time for 100 steps of RK3 flow integrator: ")
@time wfl_rk3(U, 100, 0.01, gp, lp, ymws)
eoft = Eoft_plaq(U, gp, lp, ymws)
eoft = Eoft_clover(U, gp, lp, ymws)
qtop = Qtop(U, gp, lp, ymws)

@time eoft = Eoft_plaq(U, gp, lp, ymws)
println("Plaq: ", eoft)
@time eoft = Eoft_clover(U, gp, lp, ymws)
println("Clov: ", eoft)
@time qtop = Qtop(U, gp, lp, ymws)
println("Qtop: ", qtop)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("## END Wilson action/flow measurements")

# Set gauge parameters
# SECOND SET: Improved action/flow
println("\n## IMPROVED ACTION/FLOW TIMES")
gp = GaugeParm{PREC}(GRP{PREC}, 6.0, 5/6, (0.5,0.5))
println("Gauge  Parameters: ", gp)

println("Initial Action: ")
@time S = gauge_action(U, lp, gp, ymws)


HMC!(U,int.eps,1,lp, gp, ymws, noacc=true)
pl = Vector{Float64}()
for i in 1:4
    @time dh, acc = HMC!(U,int,lp, gp, ymws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

zfl_rk3(U, 1, 0.01, gp, lp, ymws)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("Time for 100 steps of RK3 flow integrator: ")
@time zfl_rk3(U, 100, 0.01, gp, lp, ymws)
println("Action: ", gauge_action(U, lp, gp, ymws))
println("## END improved action/flow measurements")

print_timer(linechars = :ascii)
println("END")


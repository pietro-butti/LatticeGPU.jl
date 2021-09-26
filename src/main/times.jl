using CUDA, Logging, StructArrays, Random

CUDA.allowscalar(false)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

# Set lattice/block size
lp = SpaceParm{4}((96,96,32,32), (4,4,4,4))
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

println("Allocating YM workspace")
ymws = YMworkspace(GRP, PREC, lp)

# Main program
println("Allocating gauge field")
U = field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))

println("Time to take the configuration to memory: ")
@time Ucpu = Array(U)


# Set gauge parameters
# FIRST SET: Wilson action/flow
println("\n## WILSON ACTION/FLOW TIMES")
gp = GaugeParm{PREC}(6.0, 1.0, (0.0,0.0), 3)
println("Gauge  Parameters: ", gp)

println("Initial Action: ")
@time S = gauge_action(U, lp, gp, ymws)


dt = 0.1
ns  = 10
HMC!(U,dt,1,lp, gp, ymws, noacc=true)

pl = Vector{Float64}()
for i in 1:4
    @time dh, acc = HMC!(U,dt,ns,lp, gp, ymws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

wfl_rk3(U, 1, 0.01, lp, ymws)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("Time for 100 steps of RK3 flow integrator: ")
@time wfl_rk3(U, 100, 0.01, lp, ymws)
println("Action: ", gauge_action(U, lp, gp, ymws))
println("## END Wilson action/flow measurements")

# Set gauge parameters
# SECOND SET: Improved action/flow
println("\n## IMPROVED ACTION/FLOW TIMES")
gp = GaugeParm{PREC}(6.0, 5.0/3.0, (0.0,0.0), 3)
println("Gauge  Parameters: ", gp)

println("Initial Action: ")
@time S = gauge_action(U, lp, gp, ymws)


dt = 0.1
ns  = 10
HMC!(U,dt,1,lp, gp, ymws, noacc=true)

pl = Vector{Float64}()
for i in 1:4
    @time dh, acc = HMC!(U,dt,ns,lp, gp, ymws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

zfl_rk3(U, 1, 0.01, lp, ymws)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("Time for 100 steps of RK3 flow integrator: ")
@time zfl_rk3(U, 100, 0.01, lp, ymws)
println("Action: ", gauge_action(U, lp, gp, ymws))
println("## END improved action/flow measurements")

println("END")


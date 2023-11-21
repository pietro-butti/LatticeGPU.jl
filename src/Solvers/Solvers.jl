###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Solvers.jl
### created: Tue Nov 30 11:09:51 2021
###                               

module Solvers

using CUDA, TimerOutputs
using ..Space
using ..Groups
using ..Fields
using ..YM
using ..Spinors
using ..Dirac

include("CG.jl")
export CG!

include("Propagators.jl")
export propagator!, bndpropagator!, Tbndpropagator!, bndtobnd


end

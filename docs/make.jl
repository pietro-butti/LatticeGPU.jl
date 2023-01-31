using Documenter

import Pkg
Pkg.activate("../")
using LatticeGPU

makedocs(sitename="LatticeGPU", modules=[LatticeGPU], doctest=true,
         repo = "https://igit.ific.uv.es/alramos/latticegpu.jl")

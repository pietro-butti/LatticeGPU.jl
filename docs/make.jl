using Documenter

import Pkg
Pkg.activate("../")
using LatticeGPU

makedocs(sitename="LatticeGPU", modules=[LatticeGPU], doctest=true,
         pages = [
             "LatticeGPU.jl" => "index.md", 
             "Space-time" => "space.md",
             "Groups and algebras" => "groups.md",
             "Fields" => "fields.md"
             "Dirac" => "dirac.md"
             ], 
         repo = "https://igit.ific.uv.es/alramos/latticegpu.jl")

###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Spinor.jl
### created: Wed Nov 17 13:00:31 2021
###                               

module Spinors

using ..Groups
import ..Groups.imm, ..Groups.mimm, ..Groups.norm, ..Groups.norm2

struct Spinor{NS,G}
    s::NTuple{NS,G}
end
Base.zero(::Type{Spinor{NS,G}}) where {NS,G} = Spinor{NS,G}(ntuple(i->zero(G), NS))

"""
    norm2(a::Spinor)

Returns the norm of a fundamental element. Same result as sqrt(dot(a,a)).
"""
@generated function norm2(a::Spinor{NS,G}) where {NS,G}

    sum = :(norm2(a.s[1]))
    for i in 2:NS
        sum = :($sum + norm2(a.s[$i]))
    end

    return :($sum)
end

"""
    norm(a::Spinor)

Returns the norm of a fundamental element. Same result as sqrt(dot(a,a)).
"""
norm(a::Spinor{NS,G}) where {NS,G} = sqrt(norm2(a))

"""
    dot(a::Spinor,b::Spinor)

Returns the scalar product of two spinors.
"""
@generated function dot(a::Spinor{NS,G},b::Spinor{NS,G}) where {NS,G}  

    sum = :(dot(a.s[1],b.s[1]))
    for i in 2:NS
        sum = :($sum + norm2(a.s[$i]))
    end

    return :($sum)
end


"""
    *(g::SU3{T},b::Spinor)

Returns ga
"""
Base.:*(g::S,b::Spinor{NS,G}) where {S <: Group,NS,G} = Spinor{NS,G}(ntuple(i->g*b.s[i], NS))

"""
    \\(g::SU3{T},b::Spinor{NS,G})

Returns g^+ a
"""
Base.:\(g::S,b::Spinor{NS,G}) where {S <: Group,NS,G} = Spinor{NS,G}(ntuple(i->g\b.s[i], NS))


Base.:+(a::Spinor{NS,G},b::Spinor{NS,G}) where {NS,G} = Spinor{NS,G}(ntuple(i->a.s[i]+b.s[i], NS))
Base.:-(a::Spinor{NS,G},b::Spinor{NS,G}) where {NS,G} = Spinor{NS,G}(ntuple(i->a.s[i]-b.s[i], NS))
Base.:+(a::Spinor{NS,G})                 where {NS,G} = a
Base.:-(a::Spinor{NS,G})                 where {NS,G} = Spinor{NS,G}(ntuple(i->-b.s[i], NS))
imm(a::Spinor{NS,G})                     where {NS,G} = Spinor{NS,G}(ntuple(i->imm(b.s[i]), NS))
mimm(a::Spinor{NS,G})                    where {NS,G} = Spinor{NS,G}(ntuple(i->mimm(b.s[i]), NS))


# Operations with numbers
Base.:*(a::Spinor{NS,G},b::Number) where {NS,G} = Spinor{NS,G}(ntuple(i->b*a.s[i], NS))
Base.:*(b::Number,a::Spinor{NS,G}) where {NS,G} = Spinor{NS,G}(ntuple(i->b*a.s[i], NS))
Base.:/(a::Spinor{NS,G},b::Number) where {NS,G} = Spinor{NS,G}(ntuple(i->a.s[i]/b, NS))


## 
# Operations specific for Spinors in 4D
##

# dummy structs for dispatch:
# Projectors (1+s\gamma_N)
struct Pgamma{N,S}
end

"""
    pmul(Pgamma{N,S}, a::Spinor)

Returns ``(1+s\\gamma_N)a``.
"""
function pmul(::Type{Pgamma{4,1}},  a::Spinor{4,G}) where {NS,G}
    
    r1 = a.s[1]+a.s[3]
    r2 = a.s[2]+a.s[4]
    return Spinor{4,G}((r1,r2,r1,r2))
end
function pmul(::Type{Pgamma{4,-1}}, a::Spinor{4,G}) where {NS,G}
    
    r1 = a.s[1]-a.s[3]
    r2 = a.s[2]-a.s[4] 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
function pmul(::Type{Pgamma{1,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+imm(a.s[4])
    r2 = a.s[2]+imm(a.s[3])
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
function pmul(::Type{Pgamma{1,-1}}, a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-imm(a.s[4])
    r2 = a.s[2]-imm(a.s[3])
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

function pmul(::Type{Pgamma{2,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+a.s[4]
    r2 = a.s[2]-a.s[3]
    return Spinor{4,G}((r1,r2,-r2,r1))
end
function pmul(::Type{Pgamma{2,-1}}, a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-a.s[4]
    r2 = a.s[2]+a.s[3]
    return Spinor{4,G}((r1,r2,r2,-r1))
end

function pmul(::Type{Pgamma{3,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+imm(a.s[3])
    r2 = a.s[2]-imm(a.s[4])
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
function pmul(::Type{Pgamma{3,-1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-imm(a.s[3])
    r2 = a.s[2]+imm(a.s[4])
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end


"""
    gpmul(pgamma{N,S}, g::G, a::Spinor) G <: Group

Returns ``g(1+s\\gamma_N)a``
"""
function gpmul(::Type{Pgamma{4,1}},  g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g*(a.s[1]+a.s[3])
    r2 = g*(a.s[2]+a.s[4])
    return Spinor{4,G}((r1,r2,r1,r2))
end
function gpmul(::Type{Pgamma{4,-1}}, g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g*(a.s[1]-a.s[3])
    r2 = g*(a.s[2]-a.s[4]) 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
function gpmul(::Type{Pgamma{1,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+imm(a.s[4]))
    r2 = g*(a.s[2]+imm(a.s[3]))
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
function gpmul(::Type{Pgamma{1,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-imm(a.s[4]))
    r2 = g*(a.s[2]-imm(a.s[3]))
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

function gpmul(::Type{Pgamma{2,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+a.s[4])
    r2 = g*(a.s[2]-a.s[3])
    return Spinor{4,G}((r1,r2,-r2,r1))
end
function gpmul(::Type{Pgamma{2,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-a.s[4])
    r2 = g*(a.s[2]+a.s[3])
    return Spinor{4,G}((r1,r2,r2,-r1))
end

function gpmul(::Type{Pgamma{3,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+imm(a.s[3]))
    r2 = g*(a.s[2]-imm(a.s[4]))
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
function gpmul(::Type{Pgamma{3,-1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-imm(a.s[3]))
    r2 = g*(a.s[2]+imm(a.s[4]))
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end

"""
    gdagpmul(pgamma{N,S}, g::G, a::Spinor) G <: Group

Returns ``g^+ (1+s\\gamma_N)a``
"""
function gdagpmul(::Type{Pgamma{4,1}},  g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g\(a.s[1]+a.s[3])
    r2 = g\(a.s[2]+a.s[4])
    return Spinor{4,G}((r1,r2,r1,r2))
end
function gdagpmul(::Type{Pgamma{4,-1}}, g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g\(a.s[1]-a.s[3])
    r2 = g\(a.s[2]-a.s[4]) 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
function gdagpmul(::Type{Pgamma{1,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+imm(a.s[4]))
    r2 = g\(a.s[2]+imm(a.s[3]))
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
function gdagpmul(::Type{Pgamma{1,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-imm(a.s[4]))
    r2 = g\(a.s[2]-imm(a.s[3]))
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

function gdagpmul(::Type{Pgamma{2,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+a.s[4])
    r2 = g\(a.s[2]-a.s[3])
    return Spinor{4,G}((r1,r2,-r2,r1))
end
function gdagpmul(::Type{Pgamma{2,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-a.s[4])
    r2 = g\(a.s[2]+a.s[3])
    return Spinor{4,G}((r1,r2,r2,-r1))
end

function gdagpmul(::Type{Pgamma{3,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+imm(a.s[3]))
    r2 = g\(a.s[2]-imm(a.s[4]))
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
function gdagpmul(::Type{Pgamma{3,-1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-imm(a.s[3]))
    r2 = g\(a.s[2]+imm(a.s[4]))
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end

export Spinor, Pgamma
export norm, norm2, dot, imm, mimm
export pmul, gpmul, gdagpmul

end

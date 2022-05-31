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
import ..Groups.imm, ..Groups.mimm, ..Groups.norm, ..Groups.norm2, ..Groups.dot

struct Spinor{NS,G}
    s::NTuple{NS,G}
end
Base.zero(::Type{Spinor{NS,G}}) where {NS,G} = Spinor{NS,G}(ntuple(i->zero(G), NS))

"""
    norm2(a::Spinor)

Returns the norm squared of a fundamental element. Same result as `dot(a,a)`.
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
@inline function pmul(::Type{Pgamma{4,1}},  a::Spinor{4,G}) where {NS,G}
    
    r1 = a.s[1]+a.s[3]
    r2 = a.s[2]+a.s[4]
    return Spinor{4,G}((r1,r2,r1,r2))
end
@inline function pmul(::Type{Pgamma{4,-1}}, a::Spinor{4,G}) where {NS,G}
    
    r1 = a.s[1]-a.s[3]
    r2 = a.s[2]-a.s[4] 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
@inline function pmul(::Type{Pgamma{1,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+imm(a.s[4])
    r2 = a.s[2]+imm(a.s[3])
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
@inline function pmul(::Type{Pgamma{1,-1}}, a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-imm(a.s[4])
    r2 = a.s[2]-imm(a.s[3])
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

@inline function pmul(::Type{Pgamma{2,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+a.s[4]
    r2 = a.s[2]-a.s[3]
    return Spinor{4,G}((r1,r2,-r2,r1))
end
@inline function pmul(::Type{Pgamma{2,-1}}, a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-a.s[4]
    r2 = a.s[2]+a.s[3]
    return Spinor{4,G}((r1,r2,r2,-r1))
end

@inline function pmul(::Type{Pgamma{3,1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]+imm(a.s[3])
    r2 = a.s[2]-imm(a.s[4])
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
@inline function pmul(::Type{Pgamma{3,-1}},  a::Spinor{4,G}) where {NS,G}

    r1 = a.s[1]-imm(a.s[3])
    r2 = a.s[2]+imm(a.s[4])
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end


"""
    gpmul(pgamma{N,S}, g::G, a::Spinor) G <: Group

Returns ``g(1+s\\gamma_N)a``
"""
@inline function gpmul(::Type{Pgamma{4,1}},  g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g*(a.s[1]+a.s[3])
    r2 = g*(a.s[2]+a.s[4])
    return Spinor{4,G}((r1,r2,r1,r2))
end
@inline function gpmul(::Type{Pgamma{4,-1}}, g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g*(a.s[1]-a.s[3])
    r2 = g*(a.s[2]-a.s[4]) 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
@inline function gpmul(::Type{Pgamma{1,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+imm(a.s[4]))
    r2 = g*(a.s[2]+imm(a.s[3]))
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
@inline function gpmul(::Type{Pgamma{1,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-imm(a.s[4]))
    r2 = g*(a.s[2]-imm(a.s[3]))
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

@inline function gpmul(::Type{Pgamma{2,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+a.s[4])
    r2 = g*(a.s[2]-a.s[3])
    return Spinor{4,G}((r1,r2,-r2,r1))
end
@inline function gpmul(::Type{Pgamma{2,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-a.s[4])
    r2 = g*(a.s[2]+a.s[3])
    return Spinor{4,G}((r1,r2,r2,-r1))
end

@inline function gpmul(::Type{Pgamma{3,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]+imm(a.s[3]))
    r2 = g*(a.s[2]-imm(a.s[4]))
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
@inline function gpmul(::Type{Pgamma{3,-1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g*(a.s[1]-imm(a.s[3]))
    r2 = g*(a.s[2]+imm(a.s[4]))
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end

"""
    gdagpmul(pgamma{N,S}, g::G, a::Spinor) G <: Group

Returns ``g^+ (1+s\\gamma_N)a``
"""
@inline function gdagpmul(::Type{Pgamma{4,1}},  g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g\(a.s[1]+a.s[3])
    r2 = g\(a.s[2]+a.s[4])
    return Spinor{4,G}((r1,r2,r1,r2))
end
@inline function gdagpmul(::Type{Pgamma{4,-1}}, g, a::Spinor{4,G}) where {NS,G}
    
    r1 = g\(a.s[1]-a.s[3])
    r2 = g\(a.s[2]-a.s[4]) 
    return Spinor{4,G}((r1,r2,-r1,-r2))
end
    
@inline function gdagpmul(::Type{Pgamma{1,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+imm(a.s[4]))
    r2 = g\(a.s[2]+imm(a.s[3]))
    return Spinor{4,G}((r1,r2,mimm(r2),mimm(r1)))
end
@inline function gdagpmul(::Type{Pgamma{1,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-imm(a.s[4]))
    r2 = g\(a.s[2]-imm(a.s[3]))
    return Spinor{4,G}((r1,r2,imm(r2),imm(r1)))
end

@inline function gdagpmul(::Type{Pgamma{2,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+a.s[4])
    r2 = g\(a.s[2]-a.s[3])
    return Spinor{4,G}((r1,r2,-r2,r1))
end
@inline function gdagpmul(::Type{Pgamma{2,-1}}, g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-a.s[4])
    r2 = g\(a.s[2]+a.s[3])
    return Spinor{4,G}((r1,r2,r2,-r1))
end

@inline function gdagpmul(::Type{Pgamma{3,1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]+imm(a.s[3]))
    r2 = g\(a.s[2]-imm(a.s[4]))
    return Spinor{4,G}((r1,r2,mimm(r1),imm(r2)))
end
@inline function gdagpmul(::Type{Pgamma{3,-1}},  g, a::Spinor{4,G}) where {NS,G}

    r1 = g\(a.s[1]-imm(a.s[3]))
    r2 = g\(a.s[2]+imm(a.s[4]))
    return Spinor{4,G}((r1,r2,imm(r1),mimm(r2)))
end


# dummy structs for dispatch:
# Basis of \\Gamma_n
struct Gamma{N}
end

"""
    dmul(n::Int64, a::Spinor)

Returns ``\\Gamma_n a``

indexing for Dirac basis ``\\Gamma_n``:

 1  gamma1
 2  gamma2
 3  gamma3
 4  gamma0
 5  gamma5
 6  gamma1 gamma5
 7  gamma2 gamma5
 8  gamma3 gamma5
 9  gamma0 gamma5
10  sigma01
11  sigma02
12  sigma03
13  sigma12
14  sigma23
15  sigma31
16  identity

"""
@inline dmul(::Type{Gamma{1}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(mimm(a.s[4]), mimm(a.s[3]), imm(a.s[2]), imm(a.s[1]))
@inline dmul(::Type{Gamma{2}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(-a.s[4], a.s[3], a.s[2], -a.s[1])
@inline dmul(::Type{Gamma{3}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(mimm(a.s[3]), imm(a.s[4]), imm(a.s[1]), mimm(a.s[2]))
@inline dmul(::Type{Gamma{4}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(-a.s[3], -a.s[4], -a.s[1], -a.s[2])
@inline dmul(::Type{Gamma{5}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( a.s[1],  a.s[2],  -a.s[3], -a.s[4])
@inline dmul(::Type{Gamma{6}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( imm(a.s[4]),  imm(a.s[3]),   imm(a.s[2]),  imm(a.s[1]))
@inline dmul(::Type{Gamma{7}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( a.s[4], -a.s[3],   a.s[2], -a.s[1])
@inline dmul(::Type{Gamma{8}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( imm(a.s[3]), mimm(a.s[4]),   imm(a.s[1]), mimm(a.s[2]))
@inline dmul(::Type{Gamma{9}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( a.s[3],  a.s[4],  -a.s[1], -a.s[2])
@inline dmul(::Type{Gamma{10}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( a.s[2],  a.s[1],  -a.s[4], -a.s[3])
@inline dmul(::Type{Gamma{11}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(mimm(a.s[2]),  imm(a.s[1]),   imm(a.s[4]), mimm(a.s[3]))
@inline dmul(::Type{Gamma{12}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( a.s[1], -a.s[2],  -a.s[3],  a.s[4])
@inline dmul(::Type{Gamma{13}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(-a.s[1],  a.s[2],  -a.s[3],  a.s[4])
@inline dmul(::Type{Gamma{14}}, a::Spinor{4,G}) where {G} = Spinor{4,G}(-a.s[2], -a.s[1],  -a.s[4], -a.s[3])
@inline dmul(::Type{Gamma{15}}, a::Spinor{4,G}) where {G} = Spinor{4,G}( imm(a.s[2]), mimm(a.s[1]),   imm(a.s[4]), mimm(a.s[3]))
@inline dmul(::Type{Gamma{16}}, a::Spinor{4,G}) where {G} = a

export Spinor, Pgamma
export norm, norm2, dot, imm, mimm
export pmul, gpmul, gdagpmul, dmul

end

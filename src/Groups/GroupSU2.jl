###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupSU2.jl
### created: Sun Jul 11 17:23:12 2021
###                               

#
# SU(2) group elements represented trough Cayley-Dickson
#       construction
# https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction
using CUDA

import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\,Base.exp
struct SU2 <: Group
    t1::ComplexF64
    t2::ComplexF64
end
SU2()           = SU2(1.0, 0.0)
inverse(b::SU2) = SU2(conj(b.t1), -b.t2)
dag(a::SU2)     = inverse(a)
norm(a::SU2)    = sqrt(abs2(a.t1) + abs2(a.t2))
norm2(a::SU2)   = abs2(a.t1) + abs2(a.t2)
tr(g::SU2)      = complex(2.0*real(g.t1), 0.0)

"""
    function normalize(a::SU2)

Return a normalized element of `SU(2)`
"""
function normalize(a::SU2)
    dr = sqrt(abs2(a.t1) + abs2(a.t2))
    if (dr == 0.0)
        return SU2(0.0)
    end
    return SU2(a.t1/dr,a.t2/dr)
end

Base.:+(a::SU2,b::SU2) = SU2(a.t1+b.t1,a.t2+b.t2)
Base.:-(a::SU2,b::SU2) = SU2(a.t1-b.t1,a.t2-b.t2)
Base.:*(a::SU2,b::SU2) = SU2(a.t1*b.t1-a.t2*conj(b.t2),a.t1*b.t2+a.t2*conj(b.t1))
Base.:/(a::SU2,b::SU2) = SU2(a.t1*conj(b.t1)+a.t2*conj(b.t2),-a.t1*b.t2+a.t2*b.t1)
Base.:\(a::SU2,b::SU2) = SU2(conj(a.t1)*b.t1+a.t2*conj(b.t2),conj(a.t1)*b.t2-a.t2*conj(b.t1))
Base.:+(a::SU2)        = SU2(a.t1,a.t2)
Base.:-(a::SU2)        = SU2(-a.t1,-a.t2)

struct SU2alg <: Algebra
    t1::Float64
    t2::Float64
    t3::Float64
end
SU2alg(x::Real)              = SU2alg(x,0.0,0.0)
SU2alg(v::Vector)            = SU2alg(v[1],v[2],v[3])
projalg(g::SU2)              = SU2alg(imag(g.t1), real(g.t2), imag(g.t2))
dot(a::SU2alg, b::SU2alg)    = a.t1*b.t1 + a.t2*b.t2 + a.t3*b.t3
norm(a::SU2alg)              = sqrt(a.t1^2 + a.t2^2 + a.t3^2)
norm2(a::SU2alg)             = a.t1^2 + a.t2^2 + a.t3^2
Base.:+(a::SU2alg)           = SU2alg(a.t1,a.t2,a.t3)
Base.:-(a::SU2alg)           = SU2alg(-a.t1,-a.t2,-a.t3)
Base.:+(a::SU2alg,b::SU2alg) = SU2alg(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3)
Base.:-(a::SU2alg,b::SU2alg) = SU2alg(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3)

Base.:*(a::SU2alg,b::Number) = SU2alg(a.t1*b,a.t2*b,a.t3*b)
Base.:*(b::Number,a::SU2alg) = SU2alg(a.t1*b,a.t2*b,a.t3*b)
Base.:/(a::SU2alg,b::Number) = SU2alg(a.t1/b,a.t2/b,a.t3/b)


"""
    function Base.exp(a::SU2alg, t::Number=1)

Computes `exp(a)`
"""
function Base.exp(a::SU2alg)
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)
    t2 = complex(sa*a.t2,sa*a.t3)
    return SU2(t1,t2)
end

function Base.exp(a::SU2alg, t::Number)
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)
    t2 = complex(sa*a.t2,sa*a.t3)
    return SU2(t1,t2)
end


"""
    function expm(g::SU2, a::SU2alg)

Computes `exp(a)*g`

"""
function expm(g::SU2, a::SU2alg)
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)*g.t1-complex(sa*a.t2,sa*a.t3)*conj(g.t2)
    t2 = complex(ca,sa*a.t1)*g.t2+complex(sa*a.t2,sa*a.t3)*conj(g.t1)
    return SU2(t1,t2)
end

"""
    function expm(g::SU2, a::SU2alg, t::Float64)

Computes `exp(t*a)*g`

"""
function expm(g::SU2, a::SU2alg, t::Float64)
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)*g.t1-complex(sa*a.t2,sa*a.t3)*conj(g.t2)
    t2 = complex(ca,sa*a.t1)*g.t2+complex(sa*a.t2,sa*a.t3)*conj(g.t1)
    return SU2(t1,t2)
               
end

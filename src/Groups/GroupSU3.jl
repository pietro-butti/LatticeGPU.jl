###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupSU3.jl
### created: Sun Jul 11 17:23:02 2021
###                               

#
# Use memory efficient representation: Only store
# first two rows. Third row constructed on the fly.
#
# a.u31 = conj(a.u12*a.u23 - a.u13*a.u22)
# a.u32 = conj(a.u13*a.u21 - a.u11*a.u23)
# a.u33 = conj(a.u11*a.u22 - a.u12*a.u21)
#
using Random

import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\,Base.one,Base.zero
import Random.rand
struct SU3{T} <: Group
    u11::Complex{T}
    u12::Complex{T}
    u13::Complex{T}
    u21::Complex{T}
    u22::Complex{T}
    u23::Complex{T}
end
inverse(a::SU3{T})       where T <: AbstractFloat = SU3{T}(conj(a.u11),conj(a.u21),(a.u12*a.u23 - a.u13*a.u22), conj(a.u12),conj(a.u22),(a.u13*a.u21 - a.u11*a.u23))
dag(a::SU3{T})           where T <: AbstractFloat = inverse(a)
tr(a::SU3{T})            where T <: AbstractFloat = a.u11+a.u22+conj(a.u11*a.u22 - a.u12*a.u21)
Base.one(::Type{SU3{T}}) where T <: AbstractFloat = SU3{T}(one(T),zero(T),zero(T),zero(T),one(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU3{T}}) where T <: AbstractFloat = exp(SU3alg{T}(randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T)))


function Base.:*(a::SU3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = conj(b.u12*b.u23 - b.u13*b.u22)
    bu32 = conj(b.u13*b.u21 - b.u11*b.u23)
    bu33 = conj(b.u11*b.u22 - b.u12*b.u21)

    return SU3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*bu31,
                  a.u11*b.u12 + a.u12*b.u22 + a.u13*bu32, 
                  a.u11*b.u13 + a.u12*b.u23 + a.u13*bu33, 
                  a.u21*b.u11 + a.u22*b.u21 + a.u23*bu31, 
                  a.u21*b.u12 + a.u22*b.u22 + a.u23*bu32,
                  a.u21*b.u13 + a.u22*b.u23 + a.u23*bu33)
end

function Base.:/(a::SU3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = (b.u12*b.u23 - b.u13*b.u22)
    bu32 = (b.u13*b.u21 - b.u11*b.u23)
    bu33 = (b.u11*b.u22 - b.u12*b.u21)

    return SU3{T}(a.u11*conj(b.u11) + a.u12*conj(b.u12) + a.u13*conj(b.u13),
                  a.u11*conj(b.u21) + a.u12*conj(b.u22) + a.u13*conj(b.u23), 
                  a.u11*(bu31)      + a.u12*(bu32)      + a.u13*(bu33), 
                  a.u21*conj(b.u11) + a.u22*conj(b.u12) + a.u23*conj(b.u13), 
                  a.u21*conj(b.u21) + a.u22*conj(b.u22) + a.u23*conj(b.u23),
                  a.u21*(bu31)      + a.u22*(bu32)      + a.u23*(bu33))
end

function Base.:\(a::SU3{T},b::SU3{T}) where T <: AbstractFloat

    au31 = (a.u12*a.u23 - a.u13*a.u22)
    au32 = (a.u13*a.u21 - a.u11*a.u23)
    bu31 = conj(b.u12*b.u23 - b.u13*b.u22)
    bu32 = conj(b.u13*b.u21 - b.u11*b.u23)
    bu33 = conj(b.u11*b.u22 - b.u12*b.u21)

    return SU3{T}(conj(a.u11)*b.u11 + conj(a.u21)*b.u21 + (au31)*bu31,
                  conj(a.u11)*b.u12 + conj(a.u21)*b.u22 + (au31)*bu32, 
                  conj(a.u11)*b.u13 + conj(a.u21)*b.u23 + (au31)*bu33, 
                  conj(a.u12)*b.u11 + conj(a.u22)*b.u21 + (au32)*bu31, 
                  conj(a.u12)*b.u12 + conj(a.u22)*b.u22 + (au32)*bu32,
                  conj(a.u12)*b.u13 + conj(a.u22)*b.u23 + (au32)*bu33)
end

function isgroup(a::SU3{T}) where T <: AbstractFloat

    tol = 1.0E-10
    g = a/a
    if ( (abs(g.u11 - 1.0) < tol) &&
         (abs(g.u12) < tol) &&
         (abs(g.u13) < tol) &&
         (abs(g.u21) < tol) &&
         (abs(g.u22 - 1.0) < tol) &&
         (abs(g.u23) < tol) )
        return true
    else
        return false
    end
end

struct SU3alg{T} <: Algebra
    t1::T
    t2::T
    t3::T
    t4::T
    t5::T
    t6::T
    t7::T
    t8::T
end

function projalg(a::SU3{T}) where T <: AbstractFloat

    sr3ov2::T = 0.866025403784438646763723170752

    ditr = ( imag(a.u11) + imag(a.u22) + 2.0*imag(a.u11*a.u22 - a.u12*a.u21) )/3.0
    m12 = (a.u12 - conj(a.u21))/2.0
    m13 = (a.u13 - (a.u12*a.u23 - a.u13*a.u22) )/2.0
    m23 = (a.u23 - (a.u13*a.u21 - a.u11*a.u23) )/2.0

    return SU3alg{T}(imag( m12 ), imag( m13 ), imag( m23 ),
                     real( m12 ), real( m13 ), real( m23 ),
                     (imag(a.u11)-imag(a.u22))/2.0,
                     sr3ov2*(ditr))
end

dot(a::SU3alg{T},b::SU3alg{T})     where T <: AbstractFloat = a.t1*b.t1 + a.t2*b.t2 + a.t3*b.t3 + a.t4*b.t4 + a.t5*b.t5 + a.t6*b.t6 + a.t7*b.t7 + a.t8*b.t8
norm2(a::SU3alg{T})                where T <: AbstractFloat = a.t1^2 + a.t2^2 + a.t3^2 + a.t4^2 + a.t5^2 + a.t6^2 + a.t7^2 + a.t8^2
norm(a::SU3alg{T})                 where T <: AbstractFloat = sqrt(a.t1^2 + a.t2^2 + a.t3^2 + a.t4^2 + a.t5^2 + a.t6^2 + a.t7^2 + a.t8^2)
Base.zero(::Type{SU3alg{T}})       where T <: AbstractFloat = SU3alg{T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU3alg{T}}) where T <: AbstractFloat = SU3alg{T}(randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T))

Base.:+(a::SU3alg{T})              where T <: AbstractFloat = SU3alg{T}(a.t1,a.t2,a.t3,a.t4,a.t5,a.t6,a.t7,a.t8)
Base.:-(a::SU3alg{T})              where T <: AbstractFloat = SU3alg{T}(-a.t1,-a.t2,-a.t3,-a.t4,-a.t5,-a.t6,-a.t7,-a.t8)
Base.:+(a::SU3alg{T},b::SU3alg{T}) where T <: AbstractFloat = SU3alg{T}(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3,a.t4+b.t4,a.t5+b.t5,a.t6+b.t6,a.t7+b.t7,a.t8+b.t8)
Base.:-(a::SU3alg{T},b::SU3alg{T}) where T <: AbstractFloat = SU3alg{T}(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3,a.t4-b.t4,a.t5-b.t5,a.t6-b.t6,a.t7-b.t7,a.t8-b.t8)
Base.:*(a::SU3alg{T},b::Number)    where T <: AbstractFloat = SU3alg{T}(b*a.t1,b*a.t2,b*a.t3,b*a.t4,b*a.t5,b*a.t6,b*a.t7,b*a.t8)
Base.:*(b::Number,a::SU3alg{T})    where T <: AbstractFloat = SU3alg{T}(b*a.t1,b*a.t2,b*a.t3,b*a.t4,b*a.t5,b*a.t6,b*a.t7,b*a.t8)
Base.:/(a::SU3alg{T},b::Number)    where T <: AbstractFloat = SU3alg{T}(a.t1/b,a.t2/b,a.t3/b,a.t4/b,a.t5/b,a.t6/b,a.t7/b,a.t8/b)



export SU3, SU3alg, inverse, dag, tr, projalg, expm, exp, norm, norm2, isgroup

struct M3x3{T} 
    u11::Complex{T}
    u12::Complex{T}
    u13::Complex{T}
    u21::Complex{T}
    u22::Complex{T}
    u23::Complex{T}
    u31::Complex{T}
    u32::Complex{T}
    u33::Complex{T}
end

Base.:*(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*b.u31,
                                                                  a.u11*b.u12 + a.u12*b.u22 + a.u13*b.u32, 
                                                                  a.u11*b.u13 + a.u12*b.u23 + a.u13*b.u33, 
                                                                  a.u21*b.u11 + a.u22*b.u21 + a.u23*b.u31, 
                                                                  a.u21*b.u12 + a.u22*b.u22 + a.u23*b.u32,
                                                                  a.u21*b.u13 + a.u22*b.u23 + a.u23*b.u33,
                                                                  a.u31*b.u11 + a.u32*b.u21 + a.u33*b.u31, 
                                                                  a.u31*b.u12 + a.u32*b.u22 + a.u33*b.u32,
                                                                  a.u31*b.u13 + a.u32*b.u23 + a.u33*b.u33)

function Base.:*(a::SU3{T},b::M3x3{T}) where T <: AbstractFloat
    
    au31 = conj(a.u12*a.u23 - a.u13*a.u22)
    au32 = conj(a.u13*a.u21 - a.u11*a.u23)
    au33 = conj(a.u11*a.u22 - a.u12*a.u21)

    return M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*b.u31,
                   a.u11*b.u12 + a.u12*b.u22 + a.u13*b.u32, 
                   a.u11*b.u13 + a.u12*b.u23 + a.u13*b.u33, 
                   a.u21*b.u11 + a.u22*b.u21 + a.u23*b.u31, 
                   a.u21*b.u12 + a.u22*b.u22 + a.u23*b.u32,
                   a.u21*b.u13 + a.u22*b.u23 + a.u23*b.u33,
                   au31*b.u11  + au32*b.u21  + au33*b.u31, 
                   au31*b.u12  + au32*b.u22  + au33*b.u32,
                   au31*b.u13  + au32*b.u23  + au33*b.u33)
end

    
function Base.:*(a::M3x3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = conj(b.u12*b.u23 - b.u13*b.u22)
    bu32 = conj(b.u13*b.u21 - b.u11*b.u23)
    bu33 = conj(b.u11*b.u22 - b.u12*b.u21)

    return M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*bu31,
                   a.u11*b.u12 + a.u12*b.u22 + a.u13*bu32, 
                   a.u11*b.u13 + a.u12*b.u23 + a.u13*bu33, 
                   a.u21*b.u11 + a.u22*b.u21 + a.u23*bu31, 
                   a.u21*b.u12 + a.u22*b.u22 + a.u23*bu32,
                   a.u21*b.u13 + a.u22*b.u23 + a.u23*bu33,
                   a.u31*b.u11 + a.u32*b.u21 + a.u33*bu31, 
                   a.u31*b.u12 + a.u32*b.u22 + a.u33*bu32,
                   a.u31*b.u13 + a.u32*b.u23 + a.u33*bu33)
end

function Base.:/(a::M3x3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = (b.u12*b.u23 - b.u13*b.u22)
    bu32 = (b.u13*b.u21 - b.u11*b.u23)
    bu33 = (b.u11*b.u22 - b.u12*b.u21)

    return M3x3{T}(a.u11*conj(b.u11) + a.u12*conj(b.u12) + a.u13*conj(b.u13),
                   a.u11*conj(b.u21) + a.u12*conj(b.u22) + a.u13*conj(b.u23), 
                   a.u11*(bu31) + a.u12*(bu32) + a.u13*(bu33), 
                   a.u21*conj(b.u11) + a.u22*conj(b.u12) + a.u23*conj(b.u13), 
                   a.u21*conj(b.u21) + a.u22*conj(b.u22) + a.u23*conj(b.u23),
                   a.u21*(bu31) + a.u22*(bu32) + a.u23*(bu33),
                   a.u31*conj(b.u11) + a.u32*conj(b.u12) + a.u33*conj(b.u13), 
                   a.u31*conj(b.u21) + a.u32*conj(b.u22) + a.u33*conj(b.u23),
                   a.u31*(bu31) + a.u32*(bu32) + a.u33*(bu33))
end

function Base.:\(a::SU3{T},b::M3x3{T}) where T <: AbstractFloat

    au31 = (a.u12*a.u23 - a.u13*a.u22)
    au32 = (a.u13*a.u21 - a.u11*a.u23)
    au33 = (a.u11*a.u22 - a.u12*a.u21)

    return M3x3{T}(conj(a.u11)*b.u11 + conj(a.u21)*b.u21 + (au31)*b.u31,
                   conj(a.u11)*b.u12 + conj(a.u21)*b.u22 + (au31)*b.u32, 
                   conj(a.u11)*b.u13 + conj(a.u21)*b.u23 + (au31)*b.u33, 
                   conj(a.u12)*b.u11 + conj(a.u22)*b.u21 + (au32)*b.u31, 
                   conj(a.u12)*b.u12 + conj(a.u22)*b.u22 + (au32)*b.u32,
                   conj(a.u12)*b.u13 + conj(a.u22)*b.u23 + (au32)*b.u33,
                   conj(a.u13)*b.u11 + conj(a.u23)*b.u21 + (au33)*b.u31, 
                   conj(a.u13)*b.u12 + conj(a.u23)*b.u22 + (au33)*b.u32,
                   conj(a.u13)*b.u13 + conj(a.u23)*b.u23 + (au33)*b.u33)

end

Base.:*(a::Number,b::M3x3{T}) where T <: AbstractFloat  = M3x3{T}(a*b.u11, a*b.u12, a*bu13,
                                                                  a*b.u21, a*b.u22, a*bu23,
                                                                  a*b.u31, a*b.u32, a*bu33)

Base.:*(b::M3x3{T},a::Number) where T <: AbstractFloat  = M3x3{T}(a*b.u11, a*b.u12, a*bu13,
                                                                  a*b.u21, a*b.u22, a*bu23,
                                                                  a*b.u31, a*b.u32, a*bu33)

Base.:+(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11+b.u11, a.u12+b.u12, a.u13+bu13,
                                                                  a.u21+b.u21, a.u22+b.u22, a.u23+bu23,
                                                                  a.u31+b.u31, a.u32+b.u32, a.u33+bu33)

Base.:-(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11-b.u11, a.u12-b.u12, a.u13-bu13,
                                                                  a.u21-b.u21, a.u22-b.u22, a.u23-bu23,
                                                                  a.u31-b.u31, a.u32-b.u32, a.u33-bu33)

Base.:-(b::M3x3{T}) where T <: AbstractFloat            = M3x3{T}(-b.u11, -b.u12, -bu13,
                                                                  -b.u21, -b.u22, -bu23,
                                                                  -b.u31, -b.u32, -bu33)

Base.:+(b::M3x3{T}) where T <: AbstractFloat            = M3x3{T}(b.u11, b.u12, bu13,
                                                                  b.u21, b.u22, bu23,
                                                                  b.u31, b.u32, bu33)
function projalg(a::M3x3{T}) where T <: AbstractFloat

    sr3ov2::T = 0.866025403784438646763723170752

    ditr = ( imag(a.u11) + imag(a.u22) + 2.0*imag(a.u33) )/3.0
    m12 = (a.u12 - conj(a.u21))/2.0
    m13 = (a.u13 - conj(a.u31))/2.0
    m23 = (a.u23 - conj(a.u32))/2.0

    return SU3alg{T}(imag( m12 ), imag( m13 ), imag( m23 ),
                     real( m12 ), real( m13 ), real( m23 ),
                     (imag(a.u11)-imag(a.u22))/2.0,
                     sr3ov2*(ditr))
end

function alg2mat(a::SU3alg{T}) where T <: AbstractFloat

    two::T = 2.0
    rct::T = 3.46410161513775458

    x8p::T = a.t8/rct
    x7p::T = a.t7/two
    u11::Complex{T} = complex(0.0, x7p + x8p)
    u22::Complex{T} = complex(0.0,-x7p + x8p)
    u33::Complex{T} = complex(0.0,-2.0*x8p)
    u12::Complex{T} = complex(a.t4,a.t1)/two
    u13::Complex{T} = complex(a.t5,a.t2)/two
    u23::Complex{T} = complex(a.t6,a.t3)/two
    u21::Complex{T} = -conj(u12)
    u31::Complex{T} = -conj(u13)
    u32::Complex{T} = -conj(u23)
    
    return M3x3{T}(u11,u12,u13, u21,u22,u23, u31,u32,u33)
end

Base.:*(a::SU3alg,b::SU3) = alg2mat(a)*b
Base.:*(a::SU3,b::SU3alg) = a*alg2mat(b)
Base.:/(a::SU3alg,b::SU3) = alg2mat(a)/b
Base.:\(a::SU3,b::SU3alg) = a\alg2mat(b)

@inline function exp_iter(dch::Complex{T}, tch::T) where T <: AbstractFloat

    c::NTuple{22, T} = ( 1.957294106339126128e-20, 4.110317623312164853e-19,
                         8.220635246624329711e-18, 1.561920696858622643e-16,
                         2.811457254345520766e-15, 4.779477332387385293e-14,
                         7.647163731819816473e-13, 1.147074559772972473e-11,
                         1.605904383682161451e-10, 2.087675698786809894e-09,
                         2.505210838544171879e-08, 2.755731922398589067e-07,
                         2.755731922398589065e-06, 2.480158730158730158e-05,
                         1.984126984126984127e-04, 1.388888888888888888e-03,
                         8.333333333333333333e-03, 4.166666666666666666e-02,
                         1.666666666666666666e-01, 0.5, 1.0, 1.0 )
    
    q0 = complex(c[1])
    q1 = complex(0.0)
    q2 = complex(0.0)
    @inbounds for i in 2:length(c)
        qt0 = q0
        qt1 = q1
        q0  = complex(c[i]) + dch*q2
        q1  = qt0 - tch*q2
        q2  = qt1 
    end 
    
    return q0, q1, q2
end

    
function expm(g::SU3{T}, a::SU3alg{T}, t::Number) where T <: AbstractFloat

    tpw = t^2
    M = alg2mat(a)
    Msq = M*M
    dch::Complex{T} = tpw*t*(M.u11*M.u22*M.u33 + M.u13*M.u21*M.u32 +
                             M.u31*M.u12*M.u23 - M.u11*M.u23*M.u32 -
                             M.u12*M.u21*M.u33 - M.u13*M.u22*M.u31)
    tch::T = -tpw*(real(Msq.u11)+real(Msq.u22)+real(Msq.u33))/2.0

    q0, q1, q2 = exp_iter(dch, tch)
    
    q1 = t*q1
    q2 = tpw*q2

    g2 = SU3{T}(q1*M.u11 + q2*Msq.u11+q0, q1*M.u12 + q2*Msq.u12, q1*M.u13 + q2*Msq.u13,
                q1*M.u21 + q2*Msq.u21, q1*M.u22 + q2*Msq.u22+q0, q1*M.u23 + q2*Msq.u23)

    return g2*g
end

function expm(g::SU3{T}, a::SU3alg{T}) where T <: AbstractFloat

    M = alg2mat(a)
    Msq = M*M
    dch::Complex{T} = M.u11*M.u22*M.u33 + M.u13*M.u21*M.u32 +
        M.u31*M.u12*M.u23 - M.u11*M.u23*M.u32 -
        M.u12*M.u21*M.u33 - M.u13*M.u22*M.u31
    tch::T = -(real(Msq.u11)+real(Msq.u22)+real(Msq.u33))/2.0
    
    q0, q1, q2 = exp_iter(dch, tch)
    
    g2 = SU3{T}(q1*M.u11 + q2*Msq.u11+q0, q1*M.u12 + q2*Msq.u12, q1*M.u13 + q2*Msq.u13,
                q1*M.u21 + q2*Msq.u21, q1*M.u22 + q2*Msq.u22+q0, q1*M.u23 + q2*Msq.u23)
    
    return g2*g
end

function Base.exp(a::SU3alg{T}) where T <: AbstractFloat

    M = alg2mat(a)
    Msq = M*M
    dch::Complex{T} = M.u11*M.u22*M.u33 + M.u13*M.u21*M.u32 +
        M.u31*M.u12*M.u23 - M.u11*M.u23*M.u32 -
        M.u12*M.u21*M.u33 - M.u13*M.u22*M.u31
    tch::T = -(real(Msq.u11)+real(Msq.u22)+real(Msq.u33))/2.0

    q0, q1, q2 = exp_iter(dch, tch)
    
    g2 = SU3{T}(q1*M.u11 + q2*Msq.u11+q0, q1*M.u12 + q2*Msq.u12, q1*M.u13 + q2*Msq.u13,
                q1*M.u21 + q2*Msq.u21, q1*M.u22 + q2*Msq.u22+q0, q1*M.u23 + q2*Msq.u23)

    return g2
end

function Base.exp(a::SU3alg{T}, t::Number) where T <: AbstractFloat

    tpw = t^2
    M = alg2mat(a)
    Msq = M*M
    dch::Complex{T} = tpw*t*(M.u11*M.u22*M.u33 + M.u13*M.u21*M.u32 +
                             M.u31*M.u12*M.u23 - M.u11*M.u23*M.u32 -
                             M.u12*M.u21*M.u33 - M.u13*M.u22*M.u31)
    tch::T = -tpw*(real(Msq.u11)+real(Msq.u22)+real(Msq.u33))/2.0

    q0, q1, q2 = exp_iter(dch, tch)
    
    q1 = t*q1
    q2 = tpw*q2

    g2 = SU3{T}(q1*M.u11 + q2*Msq.u11+q0, q1*M.u12 + q2*Msq.u12, q1*M.u13 + q2*Msq.u13,
                q1*M.u21 + q2*Msq.u21, q1*M.u22 + q2*Msq.u22+q0, q1*M.u23 + q2*Msq.u23)

    return g2
end


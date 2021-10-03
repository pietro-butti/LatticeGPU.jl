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

inverse(a::SU3{T})       where T <: AbstractFloat = SU3{T}(conj(a.u11),conj(a.u21),(a.u12*a.u23 - a.u13*a.u22), conj(a.u12),conj(a.u22),(a.u13*a.u21 - a.u11*a.u23))
dag(a::SU3{T})           where T <: AbstractFloat = inverse(a)
tr(a::SU3{T})            where T <: AbstractFloat = a.u11+a.u22+conj(a.u11*a.u22 - a.u12*a.u21)

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


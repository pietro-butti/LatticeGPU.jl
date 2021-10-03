###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    M3x3.jl
### created: Sun Oct  3 09:03:34 2021
###                               


Base.:*(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*b.u11 + a.u12*b.u21,
                                                                  a.u11*b.u12 + a.u12*b.u22, 
                                                                  a.u21*b.u11 + a.u22*b.u21, 
                                                                  a.u21*b.u12 + a.u22*b.u22)

Base.:*(a::SU2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.t1*b.u11+a.t2*b.u21,
                                                                 a.t1*b.u12+a.t2*b.u22,
                                                                 -conj(a.t2)*b.u11+conj(a.t1)*b.u21,
                                                                 -conj(a.t2)*b.u12+conj(a.t1)*b.u22)
    
Base.:*(a::M2x2{T},b::SU2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*b.t1-a.u12*conj(b.t2),
                                                                 a.u11*b.t2+a.u12*conj(b.t1),
                                                                 a.u21*b.t1-a.u22*conj(b.t2),
                                                                 -a.u21*b.t2+a.u22*conj(b.t1))

Base.:/(a::M2x2{T},b::SU2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*conj(b.t1)-a.u12*conj(b.t2),
                                                                 a.u11*b.t2+a.u12*b.t1,
                                                                 a.u21*conj(b.t1)-a.u22*conj(b.t2),
                                                                 -a.u21*b.t2+a.u22*b.t1)

Base.:\(a::SU2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(conj(a.t1)*b.u11+a.t2*b.u21,
                                                                 conj(a.t1)*b.u12+a.t2*b.u22,
                                                                 -conj(a.t2)*b.u11+a.t1*b.u21,
                                                                 -conj(a.t2)*b.u12+a.t1*b.u22)

Base.:*(a::Number,b::M2x2{T}) where T <: AbstractFloat  = M2x2{T}(a*b.u11, a*b.u12,
                                                                  a*b.u21, a*b.u22)

Base.:*(b::M2x2{T},a::Number) where T <: AbstractFloat  = M2x2{T}(a*b.u11, a*b.u12,
                                                                  a*b.u21, a*b.u22)

Base.:+(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11+b.u11, a.u12+b.u12,
                                                                  a.u21+b.u21, a.u22+b.u22)

Base.:-(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11-b.u11, a.u12-b.u12,
                                                                  a.u21-b.u21, a.u22-b.u22)

Base.:-(b::M2x2{T}) where T <: AbstractFloat            = M2x2{T}(-b.u11, -b.u12,
                                                                  -b.u21, -b.u22)

Base.:+(b::M2x2{T}) where T <: AbstractFloat            = M2x2{T}(b.u11, b.u12,
                                                                  b.u21, b.u22)

function projalg(a::M2x2{T}) where T <: AbstractFloat

    m12 = (a.u12 - conj(a.u21))/2

    return SU2alg{T}(imag( m12 ), real( m12 ), (imag(a.u11) - imag(a.u22))/2)
end


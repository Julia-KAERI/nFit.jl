mutable struct nFit
    f::Function
    p::AbstractVector{<:Real}
    xarr::AbstractVector{<:Real}
    yarr::AbstractVector{<:Real}
    earr::Union{Nothing, AbstractVector{<:Real}}
    Jacobian::Matrix{<:Real}
    W::Union{Nothing, AbstractMatrix{<:Real}}
    MaxIter::Int
    λ::Real
    Lup::Real
    Ldown::Real
    ϵ1::Real
    ϵ2::Real
    ϵ3::Real
    ϵ4::Real
    δ::Real
    
    function nFit(f, p, xarr, yarr, earr=nothing; MaxIter = 1000, λ=1.0e-2, Lup=11.0, Ldown=9.9, ϵ1 = 1.0e-3, ϵ2=1.0e-3, ϵ3 = 1.0e-1, ϵ4=1.0e-1, δ = 1.0e-3)
        @assert length(xarr) == length(yarr)
        if earr != nothing 
            @assert length(xarr) == length(earr)
        end
        try
            f(xarr[1], p)
        catch
            error("Unexpected behavior of function and parameters")
        end
        J0 = zeros( length(xarr), length(p))
        W = (earr === nothing) ? nothing : Matrix(Diagonal(earr))

        new(f, p, xarr, yarr, earr, J0, W, MaxIter, λ, Lup, Ldown, ϵ1, ϵ2, ϵ3, ϵ4, δ )
    end
end
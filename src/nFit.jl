module nFit

include("fitter.jl")

export 
    Fitter,
    fit,
    update_jacobian!,
    curve_fit

end # module nFit

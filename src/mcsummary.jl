#
# Date created: 2023-02-13
# Author: aradclif
#
#
############################################################################################

const _probs = (0.025, 0.25, 0.5, 0.75, 0.975)


"""
    mcsummary(A::AbstractMatrix{<:Real}, p::NTuple{N, T}=(0.025, 0.25, 0.5, 0.75, 0.975);
              [dim::Integer=1], [multithreaded::Bool=true]) where {T<:Real, N}

Compute the summary of the Monte Carlo simulations, where the simulation
index corresponds to dimension `dim` and `p` is the tuple of probabilities on the
interval [0,1] corresponding to the quantile(s) of interest.

The summary consists of the mean, Monte Carlo standard error, standard deviation,
and quantiles, concatenated into a matrix, in that order.
"""
function mcsummary(A::AbstractMatrix{<:Real}, p::NTuple{N, S}=_probs; dim::Integer=1, multithreaded::Bool=true) where {N, S<:Real}

    multithreaded && return tmcsummary(A, p, dim=dim)
    dim == 1 || dim == 2 || throw(DomainError(dim, "`dim` other than 1 or 2 is not a valid reduction dimension"))
    μ = vmean(A, dims=dim)
    σ = vstd(A, dims=dim, mean=μ, corrected=false)
    iden = inv(√(size(A, dim)))
    mcse = @turbo σ .* iden
    qntls = quantiles(A, float.(p), dim=dim, multithreaded=false)
    if dim == 1
        [transpose(μ) transpose(mcse) transpose(σ) transpose(qntls)]
    else # dim == 2
        [μ mcse σ qntls]
    end
end

function tmcsummary(A::AbstractMatrix{<:Real}, p::NTuple{N, S}=_probs; dim::Integer=1) where {N, S<:Real}

    dim == 1 || dim == 2 || throw(DomainError(dim, "`dim` other than 1 or 2 is not a valid reduction dimension"))
    μ = vtmean(A, dims=dim)
    σ = vstd(A, dims=dim, mean=μ, corrected=false, multithreaded=true)
    iden = inv(√(size(A, dim)))
    mcse = @tturbo σ .* iden
    qntls = quantiles(A, float.(p), dim=dim, multithreaded=true)
    if dim == 1
        [transpose(μ) transpose(mcse) transpose(σ) transpose(qntls)]
    else # dim == 2
        [μ mcse σ qntls]
    end
end



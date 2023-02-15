#
# Date created: 2023-02-13
# Author: aradclif
#
#
############################################################################################

function _quantiles_dim2(A::AbstractMatrix{T}, p::NTuple{N, S}, multithreaded::Bool) where {N, S<:AbstractFloat} where {T<:Real}
    B = similar(A, promote_type(Float64, T), size(A, 1), N)
    if multithreaded
        @inbounds Threads.@threads for i ∈ axes(A, 1)
            B[i, :] .= quantile(view(A, i, :), p)
        end
    else
        for i ∈ axes(A, 1)
            B[i, :] .= quantile(view(A, i, :), p)
        end
    end
    B
end
function _quantiles_dim1(A::AbstractMatrix{T}, p::NTuple{N, S}, multithreaded::Bool) where {N, S<:AbstractFloat} where {T<:Real}
    B = similar(A, promote_type(Float64, T), N, size(A, 2))
    if multithreaded
        @inbounds Threads.@threads for j ∈ axes(A, 2)
            B[:, j] .= quantile(view(A, :, j), p)
        end
    else
        for j ∈ axes(A, 2)
            B[:, j] .= quantile(view(A, :, j), p)
        end
    end
    B
end
quantiles(A::AbstractMatrix, p::NTuple{N, S}; dim::Integer=1, multithreaded::Bool=true) where {N, S<:AbstractFloat} =
    dim == 1 ? _quantiles_dim1(A, p, multithreaded) : _quantiles_dim2(A, p, multithreaded)

quantiles(A::AbstractMatrix, p::Vararg{S, N}; dim::Integer=1, multithreaded::Bool=true) where {S<:AbstractFloat, N} = quantiles(A, p, dim=dim, multithreaded=multithreaded)


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



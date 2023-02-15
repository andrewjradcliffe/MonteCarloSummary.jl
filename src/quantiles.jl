#
# Date created: 2023-02-15
# Author: aradclif
#
#
############################################################################################

function _quantiles_dim1(A::AbstractMatrix{T}, p::NTuple{N, S},
                         multithreaded::Bool) where {N, S<:AbstractFloat} where {T<:Real}
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

function _quantiles_dim2(A::AbstractMatrix{T}, p::NTuple{N, S},
                         multithreaded::Bool) where {N, S<:AbstractFloat} where {T<:Real}
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

function quantiles(A::AbstractMatrix, p::NTuple{N, S}; dim::Integer=1,
                   multithreaded::Bool=true) where {N, S<:AbstractFloat}
    dim == 1 ? _quantiles_dim1(A, p, multithreaded) : _quantiles_dim2(A, p, multithreaded)
end
function quantiles(A::AbstractMatrix, p::Vararg{S, N}; dim::Integer=1,
                   multithreaded::Bool=true) where {S<:AbstractFloat, N}
    quantiles(A, p, dim=dim, multithreaded=multithreaded)
end

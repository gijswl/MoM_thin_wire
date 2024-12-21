

"""
    CalcFarField(msh, Isol, coords, f, μ, ε)

Calculate the far-field electric field at coordinates `coords` using solution `Isol` on mesh `msh`.

Higher-order quadrature appears to have no influence on the result, so a 1-point quadrature is hard-coded.
"""
function CalcFarField(msh, Isol, coords, f, μ, ε)
    c = 1 / √(μ * ε)
    ω = 2π * f
    k = ω / c

    E = zeros(Complex{Float64}, size(coords, 1), 3)

    @timeit "evaluation loop" for (i, coord) ∈ enumerate(eachrow(coords))
        r = norm(coord)
        r̂ = coord / r

        Z = zeros(Complex{Float64}, 3, size(Isol, 1))
        @timeit "loop over edges" for (j, ej) ∈ enumerate(msh.edges)
            nj = msh.nodes[ej, :]
            rj = nj[2, :] .- nj[1, :]
            ℓj = norm(rj)
            bj = msh.basis_vecs[j, :]

            rm = 0.5 * (nj[1, :] .+ nj[2, :])
            Z[:, ej] += ℓj * ((r̂ ⋅ bj) * r̂ - bj) * [0.5 0.5] * exp(1im * k * (rm ⋅ r̂))
        end

        E[i, :] = 1im * ω * μ * exp(-1im * k * r) / (4π * r) * Z * Isol
    end

    return E
end
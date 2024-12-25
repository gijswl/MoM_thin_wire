using FastGaussQuadrature
using LinearAlgebra

"""
    G(k, rm, rn)

Dyadic Green's function with wavenumber `k`.
"""
function G(k, rm, rn)
    @timeit "Green's function" begin
        r = norm(rm - rn)
        return exp(-1im * k * r) / (4π * r)
    end
end

"""
    S1(x, s, a, k)

Analytic integral ∫ f(r) G(x, r) dr on segment `(x, s)` with wire radius `a` and wavenumber `k`.
"""
function S1(x, s, a, k)
    return (1 / s * √(a^2 + (x - s)^2) - 1 / s * √(a^2 + x^2) + x / s * log((x + √(a^2 + x^2)) / (x - s + √(a^2 + (x - s)^2))) - 1im * k * s / 2) / (4π)
end

"""
    S2(x, s, a, k)

Analytic integral ∫ G(x, r) dr on segment `(x, s)` with wire radius `a` and wavenumber `k`.
"""
function S2(x, s, a, k)
    return 1 / s^2 * (log((x + √(a^2 + x^2)) / (x - s + √(a^2 + (x - s)^2))) - 1im * k * s) / (4π)
end

"""
    get_quadrature(order)

Calculate Gauss-Legendre quadrature points and weights
"""
function get_quadrature(order)
    x, w = gausslegendre(order)

    return [[x[i], w[i]] for i = eachindex(x)]
end

"""
    assemble_global(msh::Mesh, src::Vector{<:Source}, f::Real, μ::Real, ε::Real; quad_order=4)

Assemble the global system matrix corresponding to mesh `msh` and sources `src`, the frequency `f`, and medium parameters `μ`, `ε`.

The quadrature order can be set through the optional argument `quad_order`.
"""
function assemble_global(msh::Mesh, src::Vector{<:Source}, f::Real, μ::Real, ε::Real; quad_order=4)
    c = 1 / √(μ * ε)
    ω = 2π * f
    k = ω / c

    a = msh.wire_radius

    # Define electric field on each element (to calculate the source vector)
    Eel = zeros(length(msh.edges))
    for s ∈ src
        if (typeof(s) == VoltageSource)
            nm = msh.nodes[msh.edges[s.edge], :]
            rm = nm[2, :] .- nm[1, :]
            ℓm = norm(rm)
            Eel[s.edge] += s.V / ℓm
        end
    end    

    # Assemble system equation Z * I = V
    ndofs = size(msh.nodes, 1)
    Z = zeros(Complex{Float64}, ndofs, ndofs)
    V = zeros(Complex{Float64}, ndofs)

    quad = get_quadrature(quad_order)

    # Pre-compute
    ne = map(e -> nodes[e, :], msh.edges) # Nodal positions
    r = map(n -> n[2, :] - n[1, :], ne)   # Element vectors
    ℓ = map(r -> norm(r), r)              # Element length

    # Local element contributions
    Amn = [0 0; 0 0]
    Φmn = [0 0; 0 0]

    # Loop over the source elements
    @timeit "outer loop" @inbounds @views for (m, em) ∈ enumerate(msh.edges)
        # Parameters of element m
        nm = ne[m]                # Nodal positions [m1; m2]
        ℓm = ℓ[m]                 # Length of element m
        bm = msh.basis_vecs[m, :] # Basis vector on element m

        # Assemble the source vector V
        # Apply quadrature over the elements to calculate Vm = ∫ f(r) ⋅ E(r) dr
        # We assume that the electric field in vector Eel is oriented along the wire axis.
        @timeit "source" for qp ∈ quad
            ξp, wp = qp
            fm1 = 0.5 * (1 - ξp)
            fm2 = 0.5 * (1 + ξp)

            V[em] += 0.5 * ℓm * wp * [fm1; fm2] .* Eel[m]
        end

        # Loop over the receiving elements
        @timeit "inner loop" @inbounds @views for (n, en) ∈ enumerate(msh.edges)
            # Parameters of element n
            nn = ne[n]                # Nodal positions [n1; n2]
            ℓn = ℓ[n]                 # Length of element n
            bn = msh.basis_vecs[n, :] # Basis vector on element n

            bdot = bm ⋅ bn

            @timeit "element" begin
                # Reset element contribution matrix
                fill!(Amn, 0)
                fill!(Φmn, 0)

                @inbounds for qp ∈ quad
                    ξp, wp = qp

                    fm1 = 0.5 * (1 - ξp)
                    fm2 = 0.5 * (1 + ξp)
                    rp = fm1 * nm[1, :] + fm2 * nm[2, :]

                    # If the source and receive elements overlap (m == n), we evaluate 
                    # the inner integral analytically and the outer numerically.
                    if (m == n)
                        @timeit "self terms" begin
                            ζp = 0.5 * (ξp + 1) * ℓm

                            Amn += 0.5 * ℓm * wp * [fm1 fm1; fm2 fm2] * S1(ζp, ℓm, a, k)
                            Φmn += 0.5 * ℓm * wp * [1 -1; -1 1] * S2(ζp, ℓm, a, k)
                        end

                    else # Otherwise, evaluate both integrals numerically.
                        @timeit "non-self terms" begin
                            @inbounds for qq ∈ quad
                                ξq, wq = qq

                                fn1 = 0.5 * (1 - ξq)
                                fn2 = 0.5 * (1 + ξq)
                                rq = fn1 * nn[1, :] + fn2 * nn[2, :]

                                Amn += 0.25 * ℓm * ℓn * wp * wq * bdot * [(fm1*fn1) (fm1*fn2); (fm2*fn1) (fm2*fn2)] * G(k, rp, rq)
                                Φmn += 0.25 * wp * wq * [1 -1; -1 1] * G(k, rp, rq)
                            end
                        end
                    end
                end
            end
            Z[em, en] += 1im * ω * μ * Amn - 1im / (ω * ε) * Φmn
        end
    end

    # Impose a Dirichlet boundary condition I = 0 on the end-points
    for ep ∈ msh.end_points
        Z[ep, :] .= 0
        Z[ep, ep] = 1
        V[ep] = 0
    end

    return Z, V
end
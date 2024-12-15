using FastGaussQuadrature
using LinearAlgebra

function G(k, rm, rn)
    r = norm(rm .- rn)
    return exp(-1im * k * r) / (4π * r)
end

function S1(x, s, a, k)
    return (1 / s * √(a^2 + (x - s)^2) - 1 / s * √(a^2 + x^2) + x / s * log((x + √(a^2 + x^2)) / (x - s + √(a^2 + (x - s)^2))) - 1im * k * s / 2) / (4π)
end

function S2(x, s, a, k)
    return 1 / s^2 * (log((x + √(a^2 + x^2)) / (x - s + √(a^2 + (x - s)^2))) - 1im * k * s) / (4π)
end

function get_quadrature(order)
    x, w = gausslegendre(order)

    return [[x[i], w[i]] for i = eachindex(x)]
end

function assemble_global(msh::Mesh, src::Vector{<:Source}, f::Real, μ::Real, ε::Real; quad_order=4)
    c = 1 / √(μ * ε)
    ω = 2π * f
    k = ω / c

    a = msh.wire_radius

    # Define source vector
    Eel = zeros(length(msh.edges))
    for s ∈ src
        if (typeof(s) == VoltageSource)
            nm = nodes[msh.edges[s.edge]]
            rm = nm[2] .- nm[1]
            ℓm = norm(rm)
            Eel[s.edge] += s.V / ℓm
        end
    end

    # Define quadrature
    quad = get_quadrature(quad_order)

    # Assemble system equation Z * I = V
    ndofs = size(msh.nodes, 1)
    Z = zeros(Complex{Float64}, ndofs, ndofs)
    V = zeros(Complex{Float64}, ndofs)

    @timeit "outer loop" for (m, em) ∈ enumerate(msh.edges)
        nm = nodes[em]
        rm = nm[2] .- nm[1]
        ℓm = norm(rm)

        V[em] += [1; 1] * 0.5 * Eel[m] * ℓm

        for (n, en) ∈ enumerate(msh.edges)
            nn = nodes[en]
            rn = nn[2] .- nn[1]
            ℓn = norm(rn)

            Amn = [0 0; 0 0]
            Φmn = [0 0; 0 0]

            @timeit "element" for (p, qp) ∈ enumerate(quad)
                ξp, wp = qp
                rp = 0.5 * (ξp + 1) .* rm .+ nm[1]
                ζp = 0.5 * (ξp + 1) * ℓm

                fm1 = 0.5 * (1 - ξp)
                fm2 = 0.5 * (1 + ξp)

                if (m == n)
                    Amn += 0.5 * ℓm * wp * [fm1 fm1; fm2 fm2] * S1(ζp, ℓm, a, k)
                    Φmn += 0.5 * ℓm * wp * [1 -1; -1 1] * S2(ζp, ℓm, a, k)
                else
                    for (q, qq) ∈ enumerate(quad)
                        ξq, wq = qq
                        rq = 0.5 * (ξq + 1) .* rn .+ nn[1]

                        fn1 = 0.5 * (1 - ξq)
                        fn2 = 0.5 * (1 + ξq)

                        Amn += 0.25 * ℓm * ℓn * wp * wq * [(fm1*fn1) (fm1*fn2); (fm2*fn1) (fm2*fn2)] * G(k, rp, rq)
                        Φmn += 0.25 * wp * wq * [1 -1; -1 1] * G(k, rp, rq)
                    end
                end
            end
            Z[em, en] += 1im * ω * μ * Amn - 1im / (ω * ε) * Φmn
        end
    end

    for ep ∈ msh.end_points
        Z[ep, :] .= 0
        Z[ep, ep] = 1
        V[ep] = 0
    end

    return Z, V
end
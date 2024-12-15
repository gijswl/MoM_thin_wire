using GLMakie

function plot_mesh(msh)
    f = Figure()
    ax1 = Axis3(f[1, 1])

    for e ∈ msh.edges
        n1 = msh.nodes[e[1], :]
        n2 = msh.nodes[e[2], :]

        p1 = Point3f(n1[1], n1[2], n1[3])
        p2 = Point3f(n2[1], n2[2], n2[3])

        lines!(ax1, [p1, p2], color=:black)
    end

    scatter!(ax1, msh.nodes, color=:black)

    for ep ∈ msh.end_points
        n = msh.nodes[ep, :]
        scatter!(ax1, Point3f(n[1], n[2], n[3]), color=:red)
    end

    return f
end


# f2 = Figure()
# ax2 = Axis3(f2[1, 1], aspect = (1, 1, 1), limits = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))

# fbplot = zeros(size(fb))
# nplot = zeros(size(fb))
# for (i, e) ∈ enumerate(edges)
#     n1 = nodes[e[1], :]
#     n2 = nodes[e[2], :]
#     fbplot[i, :] = fb[i, :] .* norm(n1 - n2)

#     nplot[i, :] = 0.5 * (n1 + n2)
# end

# quiver!(ax2, nplot[:, 1], nplot[:, 2], nplot[:, 3], fbplot[:, 1], fbplot[:, 2], fbplot[:, 3], arrowsize = 0.01, lengthscale = 0.75)

# f2
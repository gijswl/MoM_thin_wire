using GLMakie

include("../../src/Mesh.jl")
include("../../src/Visualization.jl")

ε = 8.854e-12
μ = 4e-7 * π
c = 1 / √(μ * ε)

# Generate mesh
Nel = 35
l = (c / 144e6) * 0.479
a = 1e-3

nodes, edges = mesh_segment([-l / 2 0 0], [l / 2 0 0], Nel, 1)
end_points = get_end_points(nodes, edges)
basis_vecs = assign_basis_vector(nodes, edges)

msh = Mesh(nodes, edges, end_points, basis_vecs)

# Plot the antenna in 3D
f_msh = plot_mesh(msh)
f_msh
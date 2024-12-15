using CairoMakie
using TimerOutputs

include("../src/Mesh.jl")
include("../src/Sources.jl")
include("../src/Matrix.jl")
include("../src/Visualization.jl")


ε = 8.854e-12
μ = 4e-7 * π
c = 1 / √(μ * ε)


# Generate mesh
Nel = 15
#l = [1038, 955, 956, 932, 916, 906, 897, 891, 887] * 1e-3
#x = [0, 312, 447, 699, 1050, 1482, 1986, 2553, 3168] * 1e-3
l = [1038, 955] * 1e-3
x = [0, 312] * 1e-3
a = 3e-3

global nodes = [0 0 0]
global edges = [[]]

for (len, pos) ∈ zip(l, x)
    n, e = mesh_segment([pos -len/2 0], [pos len/2 0], Nel, size(nodes, 1))
    global nodes = vcat(nodes, n)
    global edges = vcat(edges, e)
end

nodes = nodes[2:end, :]
edges = edges[2:end]

end_points = get_end_points(nodes, edges)
basis_vecs = assign_basis_vector(nodes, edges)

msh = Mesh(nodes, edges, end_points, basis_vecs, a)

# Mid-point feed
Vin = 1
idx_mid = Integer(Nel + ceil(Nel / 2))
src1 = VoltageSource(Vin, idx_mid)

src = [src1]

reset_timer!()

# Assemble system equations
@timeit "assemble" Z, V = assemble_global(msh, src, 146e6, μ, ε)

# Solve system
@timeit "solve" Isol = Z \ V

print_timer()

e = msh.edges[src[1].edge]
Iin = sum(Isol[e]) / length(Isol[e])
Zin = Vin / Iin
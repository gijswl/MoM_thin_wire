using LinearAlgebra

struct Mesh
    nodes       # N × 3 matrix of node coordinates
    edges       # M-element vector of [n1, n2] edges
    end_points  # Vector of nodes connected to only one edge
    basis_vecs  # M × 3 matrix of basis vectors on the edges
    wire_radius # Wire radius (assumed to be equal for all edges)
end

"""
    mesh_segment(r1, r2, Nel, start_idx)

Generate nodes and edges for a segment from `r1` to `r2` with `Nel` elements.
To ensure correct connectivity when multiple segments are concatenated, `start_idx` can be set to the number of nodes already in the list.
"""
function mesh_segment(r1, r2, Nel, start_idx)
    Nn = Nel + 1
    x = range(0, 1, Nn)

    nodes = x .* (r2 .- r1) .+ r1
    edges = [[i, i + 1] for i ∈ (start_idx:start_idx+Nel-1)]

    return nodes, edges
end

"""
    get_end_points(nodes, edges)

Get the indices of nodes connected to only one edge.
These are required to enforce a zero current boundary condition.
"""
function get_end_points(nodes, edges)
    count = zeros(size(nodes, 1))
    for e ∈ edges
        count[e] += [1, 1]
    end

    return findall(count .== 1)
end

"""
    get_conn_edges(edges, node)

Return the edges connecting to `node`.
"""
function get_conn_edges(edges, node)
    conn_e = Integer[]

    for (i, e) ∈ enumerate(edges)
        if (any(e .== node))
            push!(conn_e, i)
        end
    end

    return conn_e
end

"""
    assign_basis_vector(nodes, edges)

Assign basis vectors to the edges such that conservation of current is ensured.
"""
function assign_basis_vector(nodes, edges)
    a = zeros(length(edges), 3)
    visited = zeros(Bool, size(edges))

    cur_edge = 1
    while !all(visited)
        if (visited[cur_edge])
            next_edge = findfirst(.!visited)
            print("Already visited edge ", cur_edge, ". Jumping to ", next_edge)
            cur_edge = next_edge
        else
            basis_vec, next_edge = calculate_basis_vector(nodes, edges, cur_edge, a, visited)

            a[cur_edge, :] = basis_vec
            visited[cur_edge] = true

            if (next_edge == -1)
                cur_edge = findfirst(.!visited)
            else
                cur_edge = next_edge
            end
        end
    end

    return a
end

"""
    calculate_basis_vector(nodes, edges, cur_edge, a, visited)

Calculate a basis vector and next edge based on the state of the current edge and its neighbours.
"""
function calculate_basis_vector(nodes, edges, cur_edge, a, visited)
    n = edges[cur_edge]
    neighbours1 = filter!(e -> e ≠ cur_edge, get_conn_edges(edges, n[1]))
    neighbours2 = filter!(e -> e ≠ cur_edge, get_conn_edges(edges, n[2]))

    n1 = nodes[n, :]

    if (isempty(neighbours1))
        neighbour2 = neighbours2[1]

        if (visited[neighbour2])
            n2 = nodes[edges[neighbour2], :]
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])
            e2 = (n2[2, :] - n2[1, :]) / norm(n2[2, :] - n2[1, :])

            basis_vec = -(a[neighbour2, :] ⋅ e2) * e1
            next_edge = -1
        else
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])

            basis_vec = e1
            next_edge = neighbour2
        end
    elseif (isempty(neighbours2))
        neighbour1 = neighbours1[1]

        if (visited[neighbour1])
            n2 = nodes[edges[neighbour1], :]
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])
            e2 = (n2[2, :] - n2[1, :]) / norm(n2[2, :] - n2[1, :])

            basis_vec = -(a[neighbour1, :] ⋅ e2) * e1
            next_edge = -1
        else
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])

            basis_vec = e1
            next_edge = neighbour1
        end
    else
        neighbour1 = neighbours1[1]
        neighbour2 = neighbours2[1]

        if (visited[neighbour1] && visited[neighbour2])
            n2 = nodes[edges[neighbour2], :]
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])
            e2 = (n2[2, :] - n2[1, :]) / norm(n2[2, :] - n2[1, :])

            basis_vec = -(a[neighbour2, :] ⋅ e2) * e1
            next_edge = -1
        elseif visited[neighbour1]
            n2 = nodes[edges[neighbour1], :]
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])
            e2 = (n2[2, :] - n2[1, :]) / norm(n2[2, :] - n2[1, :])

            basis_vec = -(a[neighbour1, :] ⋅ e2) * e1
            next_edge = neighbour2
        elseif visited[neighbour2]
            n2 = nodes[edges[neighbour2], :]
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])
            e2 = (n2[2, :] - n2[1, :]) / norm(n2[2, :] - n2[1, :])

            basis_vec = -(a[neighbour2, :] ⋅ e2) * e1
            next_edge = neighbour1
        else
            e1 = (n1[1, :] - n1[2, :]) / norm(n1[1, :] - n1[2, :])

            basis_vec = e1
            next_edge = neighbour2
        end
    end

    return basis_vec, next_edge
end
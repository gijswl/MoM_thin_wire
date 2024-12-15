using LinearAlgebra

struct Mesh
    nodes
    edges
    end_points
    basis_vecs
end

function mesh_segment(r1, r2, Nel, start_idx)
    Nn = Nel + 1
    x = range(0, 1, Nn)

    nodes = x .* (r2 .- r1) .+ r1
    edges = [[i, i + 1] for i ∈ (start_idx:start_idx+Nel-1)]

    return nodes, edges
end

function get_end_points(nodes, edges)
    count = zeros(size(nodes, 1))
    for e ∈ edges
        count[e] += [1, 1]
    end

    return findall(count .== 1)
end

function get_conn_edges(edges, node)
    conn_e = Integer[]

    for (i, e) ∈ enumerate(edges)
        if (any(e .== node))
            push!(conn_e, i)
        end
    end

    return conn_e
end

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
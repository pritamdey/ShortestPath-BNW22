def bellman_ford(G, src):
    dist = {n: float('inf') for n in G.nodes}
    dist[src] = 0

    # The maximum number of edges in a spanning tree is V-1
    for _ in range(G.number_of_nodes() - 1):
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if dist[u] != float("Inf") and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight

    return dist

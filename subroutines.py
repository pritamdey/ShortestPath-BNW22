import networkx as nx
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from queue import PriorityQueue


def ballout(G, v, R):
    """
    Calculates BalloutG(v, R) and boundary(BalloutG(v, R)) for a directed graph G = (V, E),
    a vertex v ∈ V, and a distance-parameter R.
    """
    ball_out = {v}
    boundary = set()
    visited = {v}
    queue = [(v, 0)]
    while queue:
        node, dist = queue.pop(0)
        if dist <= R:
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + G[node][neighbor]['weight']))
                    if dist + G[node][neighbor]['weight'] <= R:
                        ball_out.add(neighbor)
                        boundary.add((node, neighbor))
        else:
            break
    return ball_out, boundary


def ballin(G, v, R):
    ball_in = {v}
    boundary = set()
    visited = set()
    queue = [(v, 0)]
    while queue:
        u, dist = queue.pop(0)
        if dist <= R:
            if u not in visited:
                visited.add(u)
                if dist > 0:
                    ball_in.add(u)
                for neighbor in G.predecessors(u):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + G[neighbor][u]['weight']))
                        if dist + G[neighbor][u]['weight'] <= R:
                            boundary.add((neighbor, u))

    return ball_in, boundary


def LDD(G, D):
    # Set global variable n to number of vertices in G
    n = G.number_of_nodes()
    # Copy G to G0 and initialize empty set Erem for remaining edges
    G0 = G.copy()
    E_rem = set()

    if n <= 1:
        return E_rem
    # Phase 1: mark vertices as light, out-light, or heavy
    c = 3
    k = int(c * math.log(n))
    S = set(random.sample(list(G.nodes), k))

    # for s_i in S:
    #     s_i_ballout, _ = ballout(G, s_i, D / 4)
    #     s_i_ballin = ballin(G, s_i, D / 4)

    lightNodes = set([])
    for v in G.nodes():
        v_ballin, _ = ballin(G, v, D / 4)
        if len(S.intersection(v_ballin)) <= .6 * k:
            lightNodes.add(v)
            G.nodes[v]['mark'] = 'in-light'
        else:
            v_ballout, _ = ballout(G, v, D / 4)
            if len(S.intersection(v_ballout)) <= .6 * k:
                lightNodes.add(v)
                G.nodes[v]['mark'] = 'out-light'
            else:
                G.nodes[v]['mark'] = 'heavy'
    # Phase 2: carve out balls until no light vertices remain
    p = min(1, 80 * math.log2(n) / D)
    if len(lightNodes) > 0:
        while G is not None and len(lightNodes.intersection(G.nodes())) > 0:
            v = random.sample(list(lightNodes.intersection(G.nodes())), 1)[0]
            R_v = np.random.geometric(p, 1)
            ball_star, ball_star_boundary = None, None
            if G.nodes[v]['mark'] == 'in-light':
                ball_star, ball_star_boundary = ballin(G, v, R_v)
            if G.nodes[v]['mark'] == 'out-light':
                ball_star, ball_star_boundary = ballout(G, v, R_v)
            E_boundary = ball_star_boundary
            if R_v > D / 4 or len(ball_star) > .7 * n:
                E_rem = G.edges()
                return E_rem
            E_recurse = LDD(G.subgraph(ball_star).copy(), D)
            E_rem = E_rem.union(E_boundary.union(E_recurse))
            G = G.remove_nodes_from(ball_star)

    # Phase 3: Clean up

    v = random.sample(list(G0.nodes), 1)[0]
    v_ballin, _ = ballin(G0, v, D / 2)
    v_ballout, _ = ballout(G0, v, D / 2)

    if G is not None and (not v_ballin.issuperset(G.nodes) or not v_ballout.issuperset(G.nodes)):
        E_rem = G.edges
        return E_rem

    return E_rem


def ElimNeg(G):
    s = random.sample(list(G.nodes()), 1)[0]
    d = {v: 99999 for v in G.nodes()}
    d[s] = 0
    Q = {s: d[s]}
    marked = {v: False for v in G.nodes()}
    iter = 0
    while bool(Q):
        if iter > 50:
            return "Elim Neg Did not Terminate"
        # Dijkstra Phase
        while bool(Q):
            v = min(Q, key=Q.get)
            Q.pop(v, None)
            marked[v] = True
            for x in G[v]:
                if G[v][x]['weight'] >= 0:
                    if d[v] + G[v][x]['weight'] < d[x]:
                        d[x] = d[v] + G[v][x]['weight']
                        Q[x] = d[x]
        # Bellman-Ford Phase
        for v in marked.keys():
            if marked[v]:
                for x in G[v]:
                    if G[v][x]['weight'] < 0:
                        if d[v] + G[v][x]['weight'] < d[x]:
                            d[x] = d[v] + G[v][x]['weight']
                            Q[x] = d[x]
                marked[v] = False
        iter = iter + 1
    return d


def FixDAGEdges(G, P, n):
    k = len(P)
    # Line 1: Re-order P = <V0, V1, ..., V_k-1> in topological order
    G_condensed = nx.condensation(G, P)
    order = nx.topological_sort(G_condensed)
    P = [P[i] for i in order]

    # Line 2: mu_j = min negative edge weight entering Vj, or 0 if no such edge exists
    mu = [0] * k
    for u, v in G.edges():
        for j, Vj in enumerate(P):
            if (u not in Vj) and (v in Vj) and G[u][v]['weight'] < mu[j]:
                mu[j] = G[u][v]['weight']

    # Lines 3-6
    phi = [0] * n
    M = np.cumsum(mu)
    for j, Vj in enumerate(P):
        for v in Vj:
            phi[v] = M[j]
    # Line 7
    return phi


def Dijkstra(G, source):
    # Initialize shortest path tree
    shortest_tree = nx.DiGraph()
    # Initialize distances dictionary
    distances = {node: float('inf') for node in G.nodes}
    distances[source] = 0
    # Initialize priority queue
    queue = [(0, source)]
    # Dijkstra main steps
    while queue:
        current_dist, current_node = queue.pop(0)
        for neighbor in G.neighbors(current_node):
            edge_weight = G.get_edge_data(current_node, neighbor)['weight']
            new_distance = current_dist + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                queue.append((new_distance, neighbor))
                queue.sort()
                shortest_tree.add_edge(current_node, neighbor, weight=edge_weight)
    return shortest_tree

import networkx as nx
from subroutines import LDD, ElimNeg, FixDAGEdges


def ScaleDown(G, delta, B, n):
    phi2 = [0] * n
    if len(G.nodes()) == 0:
        return phi2
    G_B = G.copy()
    for u, v in G_B.edges():
        G_B[u][v]['weight'] = max(0, G[u][v]['weight'] + B)
    if delta > 2:
        d = delta / 2

        # Phase 0: Decompose V to SCCs V1, V2, . . . with weak diameter dB in G
        E_rem = LDD(G_B, d * B)
        G_B_cut_E_rem = G_B.copy()
        G_B_cut_E_rem.remove_edges_from(E_rem)
        # V = list(G_B_cut_E_rem.nodes())
        # V_rem = set([v for (_, v) in E_rem])
        SCCs = list(nx.strongly_connected_components(G_B_cut_E_rem))
        H = nx.DiGraph()
        for SCC in SCCs:
            for u in SCC:
                for v in G.successors(u):
                    if v in SCC:
                        H.add_edge(u, v, weight=G[u][v]['weight'])
        phi1 = ScaleDown(H, d, B, n)
        G_B_phi1 = G_B.copy()
        for u, v in G_B.edges():
            G_B_phi1[u][v]['weight'] = G_B[u][v]['weight'] + phi1[u] - phi1[v]
        # Phase 2: Make all edges in GB \ Erem non-negative
        G_B_phi1.remove_edges_from(E_rem)
        psi = FixDAGEdges(G_B_phi1, SCCs, n)
        phi2 = [sum(x) for x in zip(phi1, psi)]
    # Phase 3:  Make all edges in GB non-negative
    G_B_phi2 = G_B.copy()
    for u, v in G_B.edges():
        G_B_phi2[u][v]['weight'] = G_B[u][v]['weight'] + phi2[u] - phi2[v]
    psi_prime = ElimNeg(G_B_phi2)
    phi3 = phi2
    for key, value in psi_prime.items():
        phi3[key] = phi2[key] + value
    return phi3

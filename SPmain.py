import numpy as np
from ScaleDown import ScaleDown
from subroutines import Dijkstra


def SPmain(G_in, s_in):
    # Line 1: Scale up edge weights
    n = G_in.number_of_nodes()
    B = 2 * n
    G_bar = G_in.copy()
    for u, v in G_bar.edges():
        G_bar[u][v]['weight'] = G_in[u][v]['weight'] * B
    # Line 2: Round up B
    t = int(np.ceil(np.log2(B)))
    B = 2 ** t
    # Line 3: Identity price function
    phi_i = [0] * n
    # Lines 4-6: Calling ScaleDown
    G_bar_phi = G_bar.copy()
    for i in range(1, t+1):
        for u, v in G_bar_phi.edges():
            G_bar_phi[u][v]['weight'] = G_bar_phi[u][v]['weight'] + phi_i[u] - phi_i[v]
        psi_i = ScaleDown(G_bar_phi, n, B/(2 ** i), n)
        phi_i = [sum(x) for x in zip(phi_i, psi_i)]
    # Line 7: G_star
    G_star = G_bar_phi
    for u, v in G_star.edges():
        G_star[u][v]['weight'] = G_star[u][v]['weight'] + phi_i[u] - phi_i[v] + 1
    # Lines 8-9: return shortest path tree
    return Dijkstra(G_star, s_in)

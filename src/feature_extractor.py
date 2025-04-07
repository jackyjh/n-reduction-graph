import numpy as np
from itertools import combinations
from math import log


def calculate_edge_weight(G, hedges):
    max_node = max(max(hedge) for hedge in hedges)  
    adj_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)  
    
    for hedge in hedges:
        for node1, node2 in combinations(hedge, 2):
            adj_matrix[node1, node2] += 1
            adj_matrix[node2, node1] += 1
            
    return adj_matrix


def extract_features(G, hedges, adj_matrix):
    features = []
    for hedge in hedges:
        gm = calculate_geometric_mean(G, hedge, adj_matrix)
        hm = calculate_harmonic_mean(G, hedge, adj_matrix)
        am = calculate_arithmetic_mean(G, hedge, adj_matrix)
        cn, jc, aa = calculate_neighbor_features(G, hedge)
        features.append([gm, hm, am, cn, jc, aa])
    return features 


def calculate_geometric_mean(G, hedge, adj_matrix):
    weights_sum = 0
    num_pairs = 0

    for node1, node2 in combinations(hedge, 2):
        if G.has_edge(node1, node2):
            weight = adj_matrix[node1, node2]
            if weight > 0:
                weights_sum += log(weight)
                num_pairs += 1

    if num_pairs > 0:
        return np.exp(weights_sum / num_pairs)
    else:
        return 0


def calculate_harmonic_mean(G, hedge, adj_matrix):
    weights_sum = 0
    num_pairs = 0

    for node1, node2 in combinations(hedge, 2):
        if G.has_edge(node1, node2):
            weight = adj_matrix[node1, node2]
            if weight > 0:
                weights_sum += 1 / weight
                num_pairs += 1

    if num_pairs > 0:
        return num_pairs / weights_sum
    else:
        return 0


def calculate_arithmetic_mean(G, hedge, adj_matrix):
    weights_sum = 0
    num_pairs = 0

    for node1, node2 in combinations(hedge, 2):
        if G.has_edge(node1, node2):
            weight = adj_matrix[node1, node2]
            if weight > 0:
                weights_sum += weight
                num_pairs += 1

    if num_pairs > 0:
        return weights_sum / num_pairs
    else:
        return 0


def calculate_neighbor_features(G, hedge):
    common_neighbors = set(G.neighbors(next(iter(hedge))))
    for node in hedge:
        common_neighbors.intersection_update(G.neighbors(node))
    
    cn = len(common_neighbors)
    jc = cn / len(set().union(*[G.neighbors(node) for node in hedge]))
    aa = sum(1 / log(G.degree(node)) for node in common_neighbors if G.degree(node) > 1)

    return cn, jc, aa 
import random
import itertools
import networkx as nx 


split_num = 0.8
MAX_ITER = 10000


def train_test_split(hedges):
    train_data = random.sample(hedges, int(len(hedges) * split_num))
    test_data = [x for x in hedges if x not in train_data]
    return train_data, test_data


def n_reduced_hyperedges(hedges, n):
    new_hedges = []   
    original_hedges = []  
    for hedge in hedges:
        if len(hedge) > n:
            for combo in itertools.combinations(hedge, n):
                new_hedge = list(combo)
                if new_hedge not in hedge:
                    new_hedges.append(new_hedge)
                else:
                    original_hedges.append(hedge)
        else:
            original_hedges.append(hedge)  
    reduced_hedge = new_hedges + original_hedges
    return reduced_hedge


def count_edge_sizes(hedge, sample_ratio):
    sample_hedges = random.sample(hedge, int(sample_ratio * len(hedge)))
    edge_size_count = {}
    for edge in sample_hedges:
        edge_size = len(edge)
        edge_size_count[edge_size] = edge_size_count.get(edge_size, 0) + 1
    return edge_size_count
 

def is_clique(G):
    for node in G:
        neighbors = set(G.neighbors(node))
        if not neighbors.issuperset(set(G.nodes) - {node}):
            return False
    return True

def get_cliques(G, exclude, n, SIZE):
    cliques = set()
    nodes = list(G.nodes)
    n_iter = 0
    while len(cliques) < n and n_iter < MAX_ITER:
        n_iter += 1
        node = random.choice(nodes)
        neighbors = list(G.neighbors(node))
        if len(neighbors) < SIZE - 1:
            continue
        try:
            potential_clique = [node] + random.sample(neighbors, SIZE - 1)
        except ValueError:
            continue  
        subg = G.subgraph(potential_clique)
        if is_clique(subg) and tuple(sorted(potential_clique)) not in exclude:
            cliques.add(tuple(sorted(potential_clique)))
    if len(cliques) < n:
        all_cliques = []
        for clique in nx.find_cliques(G):
            if len(clique) >= SIZE:
                all_cliques.append(tuple(sorted(clique)))
        fallback_cliques = set()
        for clique in all_cliques:
            if len(clique) == SIZE:
                fallback_cliques.add(clique)
            elif len(clique) > SIZE:
                for combo in itertools.combinations(clique, SIZE):
                    fallback_cliques.add(tuple(sorted(combo)))
        fallback_cliques = {clq for clq in fallback_cliques if clq not in exclude}
        cliques = cliques.union(fallback_cliques)
        if len(cliques) > n:
            cliques = set(random.sample(list(cliques), n))
    return cliques

def generate_negative_samples(G, hedges, num_samples, SIZE):
    exclude = set(map(tuple, hedges))
    negative_samples = get_cliques(G, exclude, num_samples, SIZE)
    return negative_samples

def generate_neg_samples(G, hedge, hedge_samples):
    samples = []
    for en, em in hedge_samples.items():
        generated_samples = generate_negative_samples(G, hedge, em, en)
        samples.extend(generated_samples)
    samples = [list(t) for t in samples]
    return samples

import os
import random
from data_reader import read_hypergraph_data
from feature_extractor import calculate_edge_weight, extract_features
from hyperedge_model import train_test_split, n_reduced_hyperedges, count_edge_sizes, generate_neg_samples
from model_trainer import train_and_evaluate_model
import networkx as nx
import itertools
import numpy as np


def run_linkprediction(seed, dataset, hedge_size, IMB):
    random.seed(seed)

    time_ranges = {
        'email-Eu': (1064021027, 1064512975),
    }
    folder_path = os.path.join('dataset', dataset)
    filter_times = dataset in time_ranges
    time_range = time_ranges.get(dataset, None)

    hedges = read_hypergraph_data(folder_path, filter_times, time_range)
    G = nx.Graph()
    for hedge in hedges:
        for pair in itertools.combinations(hedge, 2):
            G.add_edge(*pair)
    adj_matrix = calculate_edge_weight(G, hedges)

    train_data, test_data = train_test_split(hedges)
    train_reduced_hedges = n_reduced_hyperedges(train_data, hedge_size)
    test_data = [t_hedge for t_hedge in test_data if len(t_hedge) <= hedge_size]

    train_neg = count_edge_sizes(train_reduced_hedges, IMB)
    train_neg_samples = generate_neg_samples(G, hedges, train_neg) 

    test_neg = count_edge_sizes(test_data, IMB) 
    test_neg_samples = generate_neg_samples(G, hedges, test_neg)
    print("Negative samples all generated!")

    rpg_features = extract_features(G, train_reduced_hedges, adj_matrix)
    ng_features = extract_features(G, train_neg_samples, adj_matrix)

    tpg_features = extract_features(G, test_data, adj_matrix)
    tng_features = extract_features(G, test_neg_samples, adj_matrix)
    print("All features extracted!")

    train_features = rpg_features + ng_features
    train_labels = [1] * len(train_reduced_hedges) + [0] * len(train_neg_samples)

    test_features = tpg_features + tng_features
    test_labels = [1] * len(test_data) + [0] * len(test_neg_samples)

    auc_roc, same_ratio = train_and_evaluate_model(train_features, train_labels, test_features, test_labels)
    print(f"AUC: {auc_roc}")
    return auc_roc, same_ratio


if __name__ == "__main__":
    datasets = ['email-Enron']
    results_folder = 'results_test'
    os.makedirs(results_folder, exist_ok=True)  

    expirement_num = 1
    hedge_size = 10
    IMB = 1.0
    seed_num = 0

    for dataset in datasets:
        with open(os.path.join(results_folder, f"{dataset}.csv"), 'w') as f:
            for nd in range(2, hedge_size + 1):
                auc_roc_values = []
                for i in range(expirement_num):
                    seed = seed_num + i
                    auc_roc, same_ratio = run_linkprediction(seed, dataset, nd, IMB)
                    auc_roc_values.append(auc_roc)
                average_auc_roc = sum(auc_roc_values) / expirement_num
                std_dev_auc_roc = np.std(auc_roc_values)
                f.write(f"{nd},{average_auc_roc}, {std_dev_auc_roc}\n")
        print("\n") 
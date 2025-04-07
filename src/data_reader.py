import os
import numpy as np
from collections import OrderedDict


def read_hypergraph_data(folder_path, filter_times, time_range):
    nverts_file = os.path.join(folder_path, 'nverts.txt')
    simplices_file = os.path.join(folder_path, 'simplices.txt')

    with open(nverts_file, 'r') as f:
        nverts_data = list(map(int, f.readlines()))

    with open(simplices_file, 'r') as f:
        simplices_data = list(map(int, f.readlines()))

    times = None
    if filter_times:
        times_file = os.path.join(folder_path, 'times.txt')
        with open(times_file, 'r') as f:
            times_data = list(map(int, f.readlines()))
        times = np.array(times_data)

    hedges = []
    index = 0
    if filter_times and time_range:
        for size, time in zip(nverts_data, times):
            if 1 < size <= 10 and time_range[0] <= time <= time_range[1]:
                hedges.append(tuple(simplices_data[index:index + size]))
            index += size
    else:
        for size in nverts_data:
            if 1 < size <= 10:
                hedges.append(tuple(simplices_data[index:index + size]))
            index += size

    hedges = list(OrderedDict.fromkeys(map(tuple, hedges)))
    
    unique_nodes = sorted(set(node for hedge in hedges for node in hedge))
    node_map = {original: new_index for new_index, original in enumerate(unique_nodes, start=1)}
    mapped_hedges = [[node_map[node] for node in hedge] for hedge in hedges]

    return mapped_hedges 
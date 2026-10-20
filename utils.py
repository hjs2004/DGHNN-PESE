import random
import numpy as np
import pandas as pd
import os
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def format_original_data_edge_index(expression_file, edge_file, mapping_file, label_file, pathway_file):
    counts_df = pd.read_table(expression_file, index_col=0)
    attributes = np.array(counts_df.values)

    genes = list(counts_df.index)
    nodes = len(genes)
    name_id_map = {}
    for index in range(nodes):
        gene = genes[index]
        genes[index] = str(gene).upper()
        name_id_map[gene] = index

    gene_name_map = {}
    gene_name_map_file = open(mapping_file, 'r').readlines()
    for line in gene_name_map_file:
        line = line.split('\t')
        gene_name_map[line[1]] = line[2]

    edge_index = []

    network_file = open(edge_file, 'r').readlines()
    for line in network_file:
        line = line.replace('\n', '').split('\t')
        if line[0] not in gene_name_map or line[1] not in gene_name_map:
            continue
        if gene_name_map[line[0]] not in genes:
            continue
        if gene_name_map[line[1]] not in genes:
            continue

        a = name_id_map[gene_name_map[line[0]]]
        b = name_id_map[gene_name_map[line[1]]]

        if a >= nodes or b >= nodes:
            print("nodes number error!")
            return None
        edge_index.append([a, b])
        edge_index.append([b, a])

    for index in range(nodes):
        edge_index.append([index, index])
        edge_index.append([index, index])

    h_edge_index = []
    count = 0
    pathway = open(pathway_file, 'r').readlines()
    for line in pathway:
        line = line.replace('\n', '').split('\t')

        for gene in line[2:]:
            if gene not in gene_name_map:
                continue
            if gene_name_map[gene] not in genes:
                continue
            a = name_id_map[gene_name_map[gene]]
            h_edge_index.append([a, count])

        count += 1

    label_file = open(label_file, 'r').readlines()
    label_1 = []
    for line in label_file:
        line = line.replace('\n', '')
        label_1.append(line.upper())

    labels = []
    for gene in genes:
        if gene in label_1:
            labels.append(1)
        else:
            labels.append(0)
    edge_index = np.array(edge_index).T
    labels = np.array(labels)
    h_edge_index = np.array(h_edge_index).T

    return attributes, labels, edge_index, h_edge_index
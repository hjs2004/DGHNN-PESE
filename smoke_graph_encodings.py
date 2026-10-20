"""Bounded real-CPDB-subgraph forward/backward check; never runs train.py.

Usage: python smoke_graph_encodings.py --nodes 1024 --epochs 2 --device cuda
No original data/checkpoint/result files are modified. This is not a benchmark
of full-graph memory or predictive quality.
"""
import argparse
import ast
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops

from graph_encodings import EncodingConfig, augment_dghnn_data
from model import DGHNN
from FT_model import FTC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if not 2 <= args.nodes <= 2048 or not 1 <= args.epochs <= 10:
        parser.error('Smoke test is limited to 2..2048 nodes and 1..10 epochs')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.manual_seed(0)
    root = Path(__file__).resolve().parent
    source = ast.parse((root / 'train.py').read_text(encoding='utf-8'))
    ordered = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'features_order' for t in n.targets))
    with h5py.File(root / 'data' / 'CPDB_multiomics.h5', 'r') as handle:
        columns = [s.decode() for s in handle['feature_names'][:]]
        raw_x = handle['features'][:args.nodes][:, [columns.index(s) for s in ordered]]
        adjacency = torch.from_numpy(handle['network'][:args.nodes, :args.nodes])
        names = np.array([[s.decode() if isinstance(s, bytes) else str(s) for s in row]
                          for row in handle['gene_names'][:args.nodes]], dtype=object)
        target = torch.from_numpy(handle['y_train'][:args.nodes]).float()
        supervised = torch.from_numpy(handle['mask_train'][:args.nodes].astype(bool)).flatten()
    if not supervised.any():
        raise RuntimeError('This prefix contains no training labels')
    edges = (adjacency > 0).nonzero().T.contiguous()
    edges, weights = add_self_loops(edges, adjacency[edges[0], edges[1]].float(), fill_value=1.)
    graph = Data(x=torch.from_numpy(raw_x).float(), edge_index=edges,
                 edge_attr=weights[:, None], node_names=names)
    mapping = {tuple(row): i for i, row in enumerate(names)}
    augmented_meta, info = augment_dghnn_data([graph], graph.x.clone(), mapping, EncodingConfig())
    batch = next(iter(DataLoader([graph], batch_size=1)))
    symbols = {row[1]: i for i, row in enumerate(names)}
    gene_symbols = {}
    with (root / 'data' / 'Homo_sapiens.gene_info').open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3:
                gene_symbols[fields[1]] = fields[2]
    incidences = []
    with (root / 'KEGG_pathway_gene.txt').open(encoding='utf-8') as handle:
        for e, line in enumerate(handle):
            for identifier in line.rstrip('\n').split('\t')[2:]:
                symbol = gene_symbols.get(identifier)
                if symbol in symbols:
                    incidences.append((symbols[symbol], e))
    hyperedges = torch.tensor(incidences, dtype=torch.long).reshape(-1, 2).T.to(args.device)
    model = DGHNN(in_features=batch.x.shape[1], depths=[3] * 9, n_hidden=[64] * 9,
        n_heads=0, num_classes=1, dropout=.1, meta_x=augmented_meta, data=batch,
        node2idx=mapping, gradient_checkpointing=True, edge_chunk_size=1024).to(args.device)
    classifier = FTC(input_dim=128, num_classes=1, dim=16, depth=4,
                     heads=4, dim_head=4, attn_drpout=.1, ff_dropout=.1).to(args.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
    use_amp = args.device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024.)
    x = batch.x.to(args.device)
    edge_index = batch.edge_index.to(args.device)
    target, supervised = target.to(args.device), supervised.to(args.device)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(nodes=args.nodes, ppi_edges=edges.shape[1],
        hypergraph_incidences=hyperedges.shape[1], device=args.device, encodings=info)), flush=True)
    for epoch in range(args.epochs):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=use_amp):
            representation = model(x, edge_index, hyperedges)
            prediction = classifier(representation[supervised])
        loss = torch.nn.functional.binary_cross_entropy(prediction.float(), target[supervised])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke-test loss')
        scaler.scale(loss).backward()
        gate_grad = model.gat.ml[0][0].edge_gate.weight.grad
        if gate_grad is None:
            raise RuntimeError('Edge encoder received no gradient')
        scaler.step(optimizer)
        scaler.update()
        if use_amp:
            torch.cuda.synchronize()
        print(json.dumps(dict(epoch=epoch, loss=loss.item(), seconds=time.perf_counter() - started,
            scaler_scale=scaler.get_scale(), edge_gate_grad_finite=bool(torch.isfinite(gate_grad).all()),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if use_amp else 0)), flush=True)
    model.eval()
    classifier.eval()
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=use_amp):
            prediction = classifier(model(x, edge_index, hyperedges))
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite evaluation predictions')
    print('SMOKE_OK', tuple(prediction.shape), flush=True)


if __name__ == '__main__':
    main()

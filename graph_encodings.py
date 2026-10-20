"""GraphGPS-inspired PE/SE preprocessing for DGHNN's shared gene coordinates.

References: GraphGPS, NeurIPS 2022, Fig. 1 / Table 1; official
graphgps/transform/posenc_stats.py. This is an independent sparse adaptation,
not a copy of the GPS architecture. No labels or omics values enter encodings.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
import torch


@dataclass(frozen=True)
class EncodingConfig:
    lap_dim: int = 8
    rw_steps: tuple = (1, 2, 3, 4, 8)
    rw_walks: int = 256
    seed: int = 0
    exact_max_nodes: int = 256

    def __post_init__(self):
        if self.lap_dim < 0 or self.rw_walks < 1 or self.exact_max_nodes < 1:
            raise ValueError('lap_dim must be nonnegative; walks and exact limit must be positive')
        if any(not isinstance(t, int) or t < 1 for t in self.rw_steps):
            raise ValueError('Random-walk steps must be positive integers')
        if len(set(self.rw_steps)) != len(self.rw_steps):
            raise ValueError('Random-walk steps must be unique')
        if self.lap_dim == 0 and not self.rw_steps:
            raise ValueError('Enable at least one of LapPE or RWSE')

    @classmethod
    def from_env(cls):
        return cls(lap_dim=int(os.environ.get('DGHNN_LAP_PE_DIM', '8')),
                   rw_steps=tuple(int(t) for t in os.environ.get(
                       'DGHNN_RW_STEPS', '1,2,3,4,8').split(',') if t.strip()),
                   rw_walks=int(os.environ.get('DGHNN_RW_WALKS', '256')),
                   seed=int(os.environ.get('DGHNN_ENCODING_SEED', '0')))


@dataclass
class GraphEncodings:
    node_pe: torch.Tensor
    node_se: torch.Tensor
    eigenvalues: torch.Tensor
    metadata: dict


def _adjacency(edge_index, num_nodes):
    edges = torch.as_tensor(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in (torch.int32, torch.int64):
        raise ValueError('edge_index must be an integer [2, E] tensor')
    if num_nodes < 1 or (edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes)):
        raise ValueError('Invalid node count or out-of-range edge index')
    rows, cols = edges.detach().cpu().numpy()
    keep = rows != cols  # Exclude artificial GCN self-loops from graph statistics.
    a = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                       (rows[keep], cols[keep])), shape=(num_nodes, num_nodes))
    a = a.maximum(a.T)
    a.data.fill(1.)  # Binary PPI topology, independent of duplicate edge counts.
    a.sort_indices()
    return a


def _lap_pe(adjacency, config):
    """Nonzero normalized-Laplacian modes, independently within each component.

    Isolates and missing frequencies are zero padded. A deterministic pivot sign
    is used; this is not SignNet and does not resolve repeated-eigenspace rotations.
    """
    n = adjacency.shape[0]
    pe = np.zeros((n, config.lap_dim), dtype=np.float32)
    values = np.zeros_like(pe)
    if not config.lap_dim:
        return pe, values
    count, labels = connected_components(adjacency, directed=False)
    rng = np.random.default_rng(config.seed)
    # Group once: avoids scanning all nodes for every isolated component.
    order = np.argsort(labels, kind='stable')
    boundaries = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    for component in range(count):
        nodes = order[boundaries[component]:boundaries[component + 1]]
        size = len(nodes)
        if size < 2:
            continue
        block = adjacency[nodes][:, nodes].astype(np.float64)
        matrix = laplacian(block, normed=True).tocsr()
        # Dense work is bounded, including when users request many frequencies.
        if size <= config.exact_max_nodes:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
        else:
            k = min(config.lap_dim + 1, size - 1)
            try:
                eigenvalues, eigenvectors = eigsh(matrix, k=k, which='SM',
                    v0=rng.standard_normal(size), tol=1e-5, maxiter=10000)
            except ArpackNoConvergence as exc:
                raise RuntimeError('Sparse LapPE did not converge; reduce DGHNN_LAP_PE_DIM '
                                   'or inspect graph connectivity') from exc
        # Each connected non-singleton component has exactly one trivial mode.
        # Drop by rank, not a float threshold that can select a rounded zero mode.
        selected = np.argsort(eigenvalues)[1:config.lap_dim + 1]
        vectors = eigenvectors[:, selected]
        if vectors.shape[1]:
            pivots = np.argmax(np.abs(vectors), axis=0)
            signs = np.sign(vectors[pivots, np.arange(vectors.shape[1])])
            vectors = vectors * np.where(signs == 0, 1, signs)
            pe[nodes, :len(selected)] = vectors
            values[nodes, :len(selected)] = eigenvalues[selected]
    return pe, values


def _rw_se(adjacency, config):
    n = adjacency.shape[0]
    output = np.zeros((n, len(config.rw_steps)), dtype=np.float32)
    if not config.rw_steps:
        return output, 'disabled'
    positions = {step: i for i, step in enumerate(config.rw_steps)}
    degree = np.diff(adjacency.indptr)
    if n <= config.exact_max_nodes:
        transition = adjacency.toarray().astype(np.float64)
        transition /= np.maximum(degree, 1)[:, None]
        power = np.eye(n)
        for step in range(1, max(config.rw_steps) + 1):
            power = power @ transition
            if step in positions:
                output[:, positions[step]] = np.diag(power)
        return output, 'exact'
    # Estimate return probabilities by seeded independent walks, not dense P^k.
    # Bound memory by 512 * rw_walks trajectories regardless of graph size.
    rng = np.random.default_rng(config.seed)
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        origins = np.repeat(np.arange(start, stop), config.rw_walks)
        current = origins.copy()
        active = degree[origins] > 0
        for step in range(1, max(config.rw_steps) + 1):
            walkers = current[active]
            offset = (rng.random(len(walkers)) * degree[walkers]).astype(np.int64)
            current[active] = adjacency.indices[adjacency.indptr[walkers] + offset]
            if step in positions:
                returned = origins[active & (current == origins)] - start
                output[start:stop, positions[step]] = np.bincount(
                    returned, minlength=stop - start) / config.rw_walks
    return output, 'monte_carlo'


def compute_encodings(edge_index, num_nodes, config=None, cache_dir=None):
    config = config or EncodingConfig()
    adjacency = _adjacency(edge_index, num_nodes)
    config_json = json.dumps(asdict(config), sort_keys=True)
    digest = hashlib.sha256(b'dghnn-pese-v2')
    digest.update(str(adjacency.shape).encode())
    digest.update(adjacency.indptr.tobytes())
    digest.update(adjacency.indices.tobytes())
    digest.update(config_json.encode())
    key = digest.hexdigest()
    cache_path = Path(cache_dir) / f'{key}.npz' if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            pe, se, values = saved['node_pe'], saved['node_se'], saved['eigenvalues']
            metadata = json.loads(str(saved['metadata'].item()))
        if (metadata.get('topology_hash') != key or pe.shape != (num_nodes, config.lap_dim)
                or se.shape != (num_nodes, len(config.rw_steps)) or values.shape != pe.shape
                or not all(np.isfinite(x).all() for x in (pe, se, values))):
            raise ValueError(f'Invalid encoding cache: {cache_path}; remove this cache entry and retry')
        metadata['cache_hit'] = True
        return GraphEncodings(torch.from_numpy(pe), torch.from_numpy(se),
                              torch.from_numpy(values), metadata)
    pe, values = _lap_pe(adjacency, config)
    se, rw_method = _rw_se(adjacency, config)
    metadata = dict(config=asdict(config), topology_hash=key, num_nodes=num_nodes,
                    undirected_edges=int(adjacency.nnz // 2), rw_method=rw_method,
                    lap_method='componentwise_normalized_nonzero', cache_hit=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.npz', delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, node_pe=pe, node_se=se, eigenvalues=values,
                                    metadata=json.dumps(metadata))
            os.replace(temporary, cache_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return GraphEncodings(torch.from_numpy(pe), torch.from_numpy(se),
                          torch.from_numpy(values), metadata)


def build_edge_features(node_pe, node_se, edge_index, raw_edge_attr=None, chunk_size=262144):
    """[raw edge features, |PE_u-PE_v|, mean(SE_u,SE_v), |SE_u-SE_v|]."""
    n = node_pe.shape[0]
    if (node_pe.ndim != 2 or node_se.ndim != 2 or node_se.shape[0] != n
            or edge_index.ndim != 2 or edge_index.shape[0] != 2 or chunk_size < 1):
        raise ValueError('Invalid node encodings, edge_index or chunk size')
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
        raise ValueError('Edge index outside node encodings')
    e = edge_index.shape[1]
    if raw_edge_attr is None:
        raw_edge_attr = node_pe.new_ones((e, 1))
    raw = torch.as_tensor(raw_edge_attr, device=node_pe.device).float()
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2 or raw.shape[0] != e or not torch.isfinite(raw).all():
        raise ValueError('raw_edge_attr must be finite and have one row per edge')
    output = node_pe.new_empty((e, raw.shape[1] + node_pe.shape[1] + 2 * node_se.shape[1]),
                               dtype=torch.float32)
    for start in range(0, e, chunk_size):
        stop = min(e, start + chunk_size)
        u, v = edge_index[:, start:stop]
        output[start:stop] = torch.cat((raw[start:stop], (node_pe[u] - node_pe[v]).abs(),
            (node_se[u] + node_se[v]) * .5, (node_se[u] - node_se[v]).abs()), dim=1)
    return output


def augment_dghnn_data(graphs, meta_x, node2idx, config=None, cache_dir=None):
    """Append shared gene-union encodings to all network copies and meta nodes.

    Preserves each graph's edge order, original node features, and raw edge attrs.
    Call once before PyG batching. Uses the existing gene identity map unchanged.
    """
    config = config or EncodingConfig()
    mappings, union_edges = [], []
    raw_dim = meta_x.shape[1]
    if len(node2idx) != len(meta_x) or set(node2idx.values()) != set(range(len(meta_x))):
        raise ValueError('node2idx must index every meta node exactly once')
    for graph in graphs:
        if getattr(graph, 'node_pe', None) is not None:
            raise ValueError('Graph already has PE/SE; do not augment twice')
        if graph.x.shape[1] != raw_dim:
            raise ValueError('All graphs and meta_x must share the original feature width')
        mapping = torch.tensor([node2idx[tuple(row)] for row in graph.node_names], dtype=torch.long)
        if len(mapping) != graph.num_nodes:
            raise ValueError('node_names must match graph nodes')
        mappings.append(mapping)
        union_edges.append(mapping[graph.edge_index.cpu()])
    if not graphs:
        raise ValueError('At least one graph is required')
    encoded = compute_encodings(torch.cat(union_edges, dim=1), len(meta_x), config, cache_dir)
    for graph, mapping in zip(graphs, mappings):
        graph.node_pe = encoded.node_pe[mapping]
        graph.node_se = encoded.node_se[mapping]
        graph.edge_attr = build_edge_features(graph.node_pe, graph.node_se, graph.edge_index,
                                              getattr(graph, 'edge_attr', None))
        raw_edge_dim = graph.edge_attr.shape[1] - config.lap_dim - 2 * len(config.rw_steps)
        graph.edge_pe = graph.edge_attr[:, raw_edge_dim:raw_edge_dim + config.lap_dim]
        graph.edge_se = graph.edge_attr[:, raw_edge_dim + config.lap_dim:]
        graph.x = torch.cat((graph.x, graph.node_pe, graph.node_se), dim=1)
    meta_augmented = torch.cat((meta_x, encoded.node_pe, encoded.node_se), dim=1)
    info = dict(encoded.metadata, enabled=True, node_raw_dim=raw_dim,
                node_output_dim=meta_augmented.shape[1], edge_output_dim=graphs[0].edge_attr.shape[1],
                graph_scope='shared_gene_union_of_all_ppi_networks')
    return meta_augmented, info

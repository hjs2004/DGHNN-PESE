import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


ROOT = Path(__file__).resolve().parents[1]


class GraphEncodingTests(unittest.TestCase):
    def encoding_module(self):
        self.assertIsNotNone(importlib.util.find_spec('graph_encodings'),
                             'The PE/SE preprocessing module must exist')
        import graph_encodings
        return graph_encodings

    def test_rwse_matches_exact_path_return_probabilities_with_isolate(self):
        enc = self.encoding_module()
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        result = enc.compute_encodings(edges, 4, enc.EncodingConfig(lap_dim=2, rw_steps=(1, 2, 3)))
        torch.testing.assert_close(result.node_se, torch.tensor([
            [0., .5, 0.], [0., 1., 0.], [0., .5, 0.], [0., 0., 0.]]))
        self.assertTrue(torch.isfinite(result.node_pe).all())
        torch.testing.assert_close(result.node_pe[3], torch.zeros(2))

    def test_laplacian_eigenvectors_satisfy_normalized_laplacian(self):
        enc = self.encoding_module()
        edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        result = enc.compute_encodings(edges, 4, enc.EncodingConfig(lap_dim=2))
        adjacency = torch.zeros(4, 4)
        adjacency[edges[0], edges[1]] = 1
        d = adjacency.sum(1).rsqrt()
        lap = torch.eye(4) - d[:, None] * adjacency * d[None, :]
        torch.testing.assert_close(lap @ result.node_pe,
                                   result.node_pe * result.eigenvalues, atol=2e-5, rtol=2e-5)

    def test_edge_encodings_preserve_order_and_undirected_symmetry(self):
        enc = self.encoding_module()
        pe = torch.tensor([[0., 2.], [3., 1.], [1., 0.]])
        se = torch.tensor([[.1], [.7], [.3]])
        edges = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
        attr = enc.build_edge_features(pe, se, edges, torch.tensor([[.2], [.2], [.8], [1.]]))
        torch.testing.assert_close(attr[0], torch.tensor([.2, 3., 1., .4, .6]))
        torch.testing.assert_close(attr[0], attr[1])
        torch.testing.assert_close(attr[3, 1:3], torch.zeros(2))

    def test_nonzero_lap_modes_exclude_degree_trivial_mode_despite_roundoff(self):
        enc = self.encoding_module()
        for seed in range(12):
            rng = np.random.default_rng(seed)
            adjacency = np.triu(rng.random((12, 12)) < .3, 1)
            adjacency[np.arange(11), np.arange(1, 12)] = True
            adjacency = adjacency | adjacency.T
            edges = torch.from_numpy(np.array(np.nonzero(adjacency)))
            result = enc.compute_encodings(edges, 12, enc.EncodingConfig(lap_dim=2))
            trivial = torch.from_numpy(np.sqrt(adjacency.sum(1))).float()
            overlap = result.node_pe.T @ trivial
            self.assertLess(overlap.abs().max().item(), 1e-4,
                            'The connected-component zero mode must never be a PE column')

    def test_cache_is_reused_and_invalidated_when_topology_changes(self):
        enc = self.encoding_module()
        cfg = enc.EncodingConfig(lap_dim=2, rw_steps=(2,))
        edges = torch.tensor([[0, 1], [1, 0]])
        with tempfile.TemporaryDirectory(prefix='dghnn-encoding-test-', dir=ROOT) as folder:
            first = enc.compute_encodings(edges, 3, cfg, cache_dir=folder)
            second = enc.compute_encodings(edges, 3, cfg, cache_dir=folder)
            third = enc.compute_encodings(torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                                          3, cfg, cache_dir=folder)
            self.assertFalse(first.metadata['cache_hit'])
            self.assertTrue(second.metadata['cache_hit'])
            self.assertNotEqual(first.metadata['topology_hash'], third.metadata['topology_hash'])
            torch.testing.assert_close(first.node_pe, second.node_pe)

    def test_sampled_walks_are_seeded_and_agree_with_exact_reference(self):
        enc = self.encoding_module()
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        cfg = enc.EncodingConfig(lap_dim=0, rw_steps=(2, 3, 4), exact_max_nodes=1,
                                 rw_walks=4096, seed=17)
        first = enc.compute_encodings(edges, 4, cfg)
        second = enc.compute_encodings(edges, 4, cfg)
        reference = enc.compute_encodings(edges, 4, enc.EncodingConfig(lap_dim=0, rw_steps=(2, 3, 4)))
        torch.testing.assert_close(first.node_se, second.node_se)
        torch.testing.assert_close(first.node_se, reference.node_se, atol=.04, rtol=0.)
        self.assertEqual(first.metadata['rw_method'], 'monte_carlo')

    def test_augmentation_shares_gene_coordinates_across_networks_and_backpropagates(self):
        enc = self.encoding_module()
        from model import DGHNN
        from FT_model import FTC
        a, b, c = ('id1', 'A'), ('id2', 'B'), ('id3', 'C')
        graphs = [
            Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]),
                 edge_attr=torch.tensor([[.5], [.5], [1.], [1.]]),
                 node_names=np.array([a, b], dtype=object)),
            Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]),
                 edge_attr=torch.ones(4, 1), node_names=np.array([b, c], dtype=object)),
        ]
        raw = [g.x.clone() for g in graphs]
        meta = torch.randn(3, 4)
        cfg = enc.EncodingConfig(lap_dim=2, rw_steps=(1, 2, 3))
        augmented, info = enc.augment_dghnn_data(graphs, meta, {a: 0, b: 1, c: 2}, cfg)
        self.assertEqual(augmented.shape, (3, 9))
        torch.testing.assert_close(graphs[0].x[:, :4], raw[0])
        torch.testing.assert_close(graphs[0].x[1, 4:], graphs[1].x[0, 4:])
        torch.testing.assert_close(augmented[1, 4:], graphs[0].x[1, 4:])
        self.assertEqual(graphs[0].edge_attr.shape, (4, 9))
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        model = DGHNN(in_features=9, depths=[1], n_hidden=[4], n_heads=0,
                      num_classes=1, dropout=0., meta_x=augmented, data=batch,
                      node2idx={a: 0, b: 1, c: 2}, gradient_checkpointing=True,
                      edge_chunk_size=2)
        classifier = FTC(input_dim=8, num_classes=1, dim=4, depth=1,
                         heads=1, dim_head=4, attn_drpout=0., ff_dropout=0.)
        output = classifier(model(batch.x, batch.edge_index, torch.tensor([[0, 1, 2], [0, 0, 0]])))
        output.square().sum().backward()
        self.assertEqual(output.shape, (3, 1))
        self.assertTrue(torch.isfinite(output).all())
        gate = model.gat.ml[0][0].edge_gate
        self.assertIsNotNone(gate.weight.grad)
        self.assertGreater(gate.weight.grad.abs().sum().item(), 0.)
        self.assertNotIn('ppi_edge_attr', model.state_dict())
        self.assertEqual(info['node_output_dim'], 9)

    def test_invalid_config_and_out_of_range_edges_fail_loudly(self):
        enc = self.encoding_module()
        with self.assertRaises(ValueError):
            enc.EncodingConfig(rw_steps=(0,))
        with self.assertRaises(ValueError):
            enc.compute_encodings(torch.tensor([[0, 3], [1, 0]]), 3, enc.EncodingConfig())


class EdgeConditioningTests(unittest.TestCase):
    def layer_class(self):
        import model
        self.assertTrue(hasattr(model, 'EdgeConditionedGCNConv'),
                        'Edge features must participate in message passing')
        return model.EdgeConditionedGCNConv

    def test_neutral_edge_gate_matches_gcn_and_nontrivial_features_change_output(self):
        layer = self.layer_class()(2, 2, edge_dim=1, edge_chunk_size=2)
        reference = GCNConv(2, 2, add_self_loops=False)
        with torch.no_grad():
            layer.lin.weight.copy_(reference.lin.weight)
            layer.bias.copy_(reference.bias)
        x = torch.tensor([[1., 0.], [0., 2.], [2., 3.]])
        edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])
        attr = torch.zeros(7, 1)
        torch.testing.assert_close(layer(x, edges, attr), reference(x, edges), atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            layer.edge_gate.weight.fill_(1.)
        changed = attr.clone()
        changed[0] = 2.
        self.assertFalse(torch.allclose(layer(x, edges, changed), layer(x, edges, attr)))
        perm = torch.tensor([2, 0, 6, 1, 4, 5, 3])
        torch.testing.assert_close(layer(x, edges[:, perm], changed[perm]),
                                   layer(x, edges, changed), atol=1e-6, rtol=1e-5)

    def test_message_chunks_do_not_retain_all_edge_by_hidden_activations(self):
        layer = self.layer_class()(8, 8, edge_dim=3, edge_chunk_size=17)
        x = torch.randn(37, 8, requires_grad=True)
        edges = torch.randint(0, 37, (2, 120))
        attributes = torch.randn(120, 3)
        saved_shapes = []
        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            output = layer(x, edges, attributes)
        self.assertNotIn((17, 8), saved_shapes,
                         'Gathered edge activations must be recomputed, not retained for every chunk')
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(layer.edge_gate.weight.grad).all())


if __name__ == '__main__':
    unittest.main()

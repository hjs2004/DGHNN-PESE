import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


ROOT = Path(__file__).resolve().parents[1]


class RuntimeFixTests(unittest.TestCase):
    def test_train_config_exposes_edge_chunk_size_override(self):
        source = (ROOT / 'train.py').read_text(encoding='utf-8')

        self.assertIn('DGHNN_EDGE_CHUNK_SIZE', source)
        self.assertIn('edge_chunk_size=EDGE_CHUNK_SIZE', source)

    def test_feature_order_is_explicit_and_matches_dataset(self):
        source = (ROOT / 'train.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'features_order'
                    for target in node.targets)
        )

        self.assertIsInstance(assignment.value, ast.List)
        feature_order = []
        for element in assignment.value.elts:
            self.assertIsInstance(element, ast.Constant)
            self.assertIsInstance(element.value, str)
            feature_order.append(element.value)

        with h5py.File(ROOT / 'data' / 'CPDB_multiomics.h5', 'r') as handle:
            dataset_features = {
                value.decode('utf-8') if isinstance(value, bytes) else value
                for value in handle['feature_names'][:]
            }

        self.assertEqual(len(feature_order), 64)
        self.assertEqual(set(feature_order), dataset_features)

    def test_train_configuration_uses_real_local_paths(self):
        source = (ROOT / 'train.py').read_text(encoding='utf-8')

        self.assertNotIn('from troch import nn', source)
        self.assertNotIn('/xxxxxxxx/', source)
        self.assertNotIn('/xxxxxxxxxxxxxx/', source)
        self.assertIn('PCNET_multiomics.h5', source)
        self.assertIn('torch.autocast', source)
        self.assertIn('GradScaler', source)
        self.assertIn('retain_graph = inner_epoch < training_times - 1', source)
        self.assertIn('output.detach().float().cpu()', source)
        self.assertIn('loss = criterion(output.float(), target_)', source)
        self.assertIn('dtype=INPUT_DTYPE', source)

        expected_files = [
            ROOT / 'data' / 'CPDB_multiomics.h5',
            ROOT / 'data' / 'IREF_multiomics.h5',
            ROOT / 'data' / 'IREF_2015_multiomics.h5',
            ROOT / 'data' / 'MULTINET_multiomics.h5',
            ROOT / 'data' / 'PCNET_multiomics.h5',
            ROOT / 'data' / 'STRINGdb_multiomics.h5',
            ROOT / 'data' / 'Homo_sapiens.gene_info',
            ROOT / 'KEGG_pathway_gene.txt',
        ]
        self.assertTrue(all(path.is_file() for path in expected_files))

    def test_dghnn_runs_on_cpu_with_batched_node_names(self):
        from model import DGHNN

        data_sets = [
            Data(
                x=torch.randn(2, 4),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                y=torch.zeros(2, 1),
                node_names=np.array([['g1', '1'], ['g2', '2']], dtype=object),
            )
        ]
        batch = next(iter(DataLoader(data_sets, batch_size=1)))
        node2idx = {('g1', '1'): 0, ('g2', '2'): 1}
        meta_x = torch.randn(2, 4)
        h_edge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)

        network = DGHNN(
            in_features=4,
            depths=[1],
            n_hidden=[2],
            n_heads=0,
            num_classes=1,
            dropout=0.0,
            gradient_checkpointing=True,
            meta_x=meta_x,
            data=batch,
            node2idx=node2idx,
        ).to('cpu')
        network.train()
        output = network(batch.x, batch.edge_index, h_edge_index)
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(network.meta_x.device.type, 'cpu')
        self.assertEqual(network.meta_edge_index.device.type, 'cpu')

    def test_chunked_gcn_matches_pyg_gcn(self):
        from model import ChunkedGCNConv

        torch.manual_seed(7)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 3, 4, 5, 0], [1, 0, 3, 2, 4, 3, 0, 5]],
            dtype=torch.long,
        )
        x = torch.randn(6, 4)
        reference = GCNConv(4, 3)
        chunked = ChunkedGCNConv(4, 3, edge_chunk_size=2)
        chunked.load_state_dict(reference.state_dict())

        reference_output = reference(x, edge_index)
        chunked_output = chunked(x, edge_index)
        torch.testing.assert_close(chunked_output, reference_output, rtol=1e-5, atol=1e-6)

    def test_training_gcn_caches_normalized_edges(self):
        from model import GCN

        network = GCN(
            in_features=4,
            depths=[1],
            n_hidden=[3],
            n_heads=0,
            num_classes=1,
            dropout=0.0,
        )

        self.assertTrue(network.ml[0][0].cached)

    def test_ft_transformer_skips_attention_stack_when_not_requested(self):
        from FT_transformer import FTTransformer

        network = FTTransformer(
            input_dim=4,
            dim=4,
            depth=2,
            heads=2,
            dim_head=2,
        )

        with patch.object(network.transformer, 'forward', wraps=network.transformer.forward) as forward:
            output = network(torch.randn(3, 4))

        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertFalse(forward.call_args.kwargs['return_attn'])

if __name__ == '__main__':
    unittest.main()

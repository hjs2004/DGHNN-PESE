import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import numpy as np


class EdgeConditionedGCNConv(nn.Module):
    """Chunked GCN with a trainable, positive edge-feature message gate.

    g_uv = 2 sigmoid(Linear(edge_attr_uv)); at zero initialization g_uv = 1.
    The topology normalization stays fixed, while the gate remains differentiable.
    Edges/self-loops must already align with edge_attr (no implicit insertion).
    """
    def __init__(self, in_channels, out_channels, edge_dim, edge_chunk_size=262144):
        super().__init__()
        if edge_dim < 1 or edge_chunk_size < 1:
            raise ValueError('edge_dim and edge_chunk_size must be positive')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.edge_gate = nn.Linear(edge_dim, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)
        self.edge_chunk_size = edge_chunk_size
        self.register_buffer('_edge_norm', None, persistent=False)
        self._norm_key = None

    def forward(self, x, edge_index, edge_attr):
        if (edge_attr is None or edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]
                or edge_attr.shape[1] != self.edge_gate.in_features):
            raise ValueError('Edge-conditioned GCN requires aligned [E, edge_dim] edge features')
        # Identity, in-place version and device detect replacement/reordering or edits.
        key = (edge_index.data_ptr(), edge_index._version, tuple(edge_index.shape),
               x.shape[0], edge_index.device)
        if key != self._norm_key:
            _, norm = gcn_norm(edge_index, num_nodes=x.shape[0], add_self_loops=False,
                               dtype=torch.float32)
            self._edge_norm = norm
            self._norm_key = key
        transformed = self.lin(x)
        output = transformed.new_zeros((x.shape[0], transformed.shape[1]))
        for start in range(0, edge_index.shape[1], self.edge_chunk_size):
            stop = min(edge_index.shape[1], start + self.edge_chunk_size)
            if self.training and torch.is_grad_enabled():
                # Without this, autograd retains gathered E*hidden activations
                # across ALL chunks. Inputs here share their original storage.
                contribution = checkpoint(self._aggregate_chunk, transformed, edge_index, edge_attr,
                    self._edge_norm, start, stop, use_reentrant=False, preserve_rng_state=False)
                output = output + contribution
            else:
                message = self._message(transformed, edge_index, edge_attr,
                                        self._edge_norm, start, stop)
                output.index_add_(0, edge_index[1, start:stop], message)
        return output + self.bias

    def _message(self, transformed, edge_index, edge_attr, norm, start, stop):
        # Keep the tiny edge encoder in FP32 rather than retaining a half copy of
        # static edge descriptors in every one of the 27 GCN layers.
        with torch.autocast(device_type=transformed.device.type, enabled=False):
            gate = 2. * torch.sigmoid(self.edge_gate(edge_attr[start:stop].float())).flatten()
            weight = norm[start:stop] * gate
        return transformed[edge_index[0, start:stop]] * weight.to(transformed.dtype)[:, None]

    def _aggregate_chunk(self, transformed, edge_index, edge_attr, norm, start, stop):
        # Include scatter inside checkpoint: index_add's backward otherwise saves
        # its entire [chunk, hidden] input even if message creation is checkpointed.
        message = self._message(transformed, edge_index, edge_attr, norm, start, stop)
        contribution = transformed.new_zeros(transformed.shape)
        return contribution.index_add_(0, edge_index[1, start:stop], message)


class ChunkedGCNConv(GCNConv):
    """GCNConv 的变体，用来限制边消息的临时张量大小。

    PyG 标准的消息传递会一次性收集所有边的源节点特征。
    在 DGHNN 的完整图上，这会创建一个数 GB 的临时张量，
    还可能触发 Windows WDDM 超时。
    把边分块聚合，既保留 GCN 的计算逻辑，又让每次 kernel 很小。
    """

    def __init__(self, *args, edge_chunk_size=262144, **kwargs):
        super().__init__(*args, **kwargs)
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_chunk_size = edge_chunk_size

    def forward(self, x, edge_index, edge_weight=None):
        # 如果 edge_index 不是 Tensor，回退到父类实现
        if not isinstance(edge_index, torch.Tensor):
            return super().forward(x, edge_index, edge_weight)

        # 归一化边权重（GCN 的标准操作）
        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops, self.flow, x.dtype,
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache

        # 线性变换
        x = self.lin(x)

        # 根据 flow 方向确定源节点和目标节点
        if self.flow == "source_to_target":
            source, target = edge_index[0], edge_index[1]
        else:
            source, target = edge_index[1], edge_index[0]

        # 分块聚合
        out = x.new_zeros((x.size(0), self.out_channels))
        for start in range(0, source.numel(), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, source.numel())
            # 取当前块的源节点特征
            message = x.index_select(self.node_dim, source[start:stop])
            # 乘边权重
            if edge_weight is not None:
                message = message * edge_weight[start:stop].view(-1, 1).to(dtype=x.dtype)
            # 累加到目标节点
            out.index_add_(self.node_dim, target[start:stop], message)

        # 加偏置
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """普通图卷积网络，用于处理 PPI 网络。

    结构：多个「卷积模块」串联，每个模块包含若干层图卷积 + 跳跃残差连接。
    每个模块结束后，把模块输入和模块输出相加（残差连接），缓解深层网络的梯度消失。
    """
    def __init__(self,
                 in_features,       # 输入特征维度，比如 64
                 depths,            # 每个模块的卷积层数，比如 [3, 3, 3, ...]
                 n_hidden,          # 每个模块的输出维度，比如 [64, 64, ...]
                 n_heads,           # GAT 的头数。0 表示用 GCNConv
                 num_classes,       # 分类数（这里没用，保留兼容）
                 concat=False,      # GAT 是否拼接多头
                 dropout=0.5,       # dropout 概率
                 leaky_relu_slope=0.2,     # LeakyReLU 的负斜率
                 gradient_checkpointing=False,  # 是否启用梯度检查点
                 edge_chunk_size=262144,        # 边分块大小
                 edge_dim=0,
    ):
        super(GCN, self).__init__()
        self.ml = nn.ModuleList()      # 每个模块的卷积层列表
        self.res = nn.ModuleList()     # 每个模块的残差连接

        in_nodes = in_features
        for index in range(len(n_hidden)):
            res_layers = nn.ModuleList()
            nodes = n_hidden[index]
            feature_in = in_nodes

            # 构建当前模块的所有卷积层
            for layer in range(depths[index]):
                if n_heads == 0:
                    # n_heads=0：用 ChunkedGCNConv
                    if edge_dim:
                        layer_temp = EdgeConditionedGCNConv(feature_in, nodes, edge_dim,
                                                            edge_chunk_size=edge_chunk_size)
                    else:
                        layer_temp = ChunkedGCNConv(
                            in_channels=feature_in, out_channels=nodes,
                            edge_chunk_size=edge_chunk_size,
                            cached=True,
                        )
                else:
                    # n_heads>0：用 GATConv
                    layer_temp = GATConv(in_channels=feature_in, out_channels=nodes,
                                         heads=n_heads, concat=concat, dropout=dropout,
                                         edge_dim=edge_dim or None)
                feature_in = nodes
                res_layers.append(layer_temp)

            # 残差连接：如果输入输出维度相同，用 Identity；否则用 Linear 投影
            if in_nodes == nodes:
                temp_res = nn.Identity()
            else:
                temp_res = nn.Linear(in_nodes, nodes)

            in_nodes = nodes
            self.ml.append(res_layers)
            self.res.append(temp_res)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.edge_dim = edge_dim

    def _forward_block(self, x, edge_index, index, edge_attr=None):
        """一个模块的前向传播：若干层卷积 + 残差连接。"""
        x_res = self.res[index](x)
        for layer in self.ml[index]:
            x = layer(x, edge_index, edge_attr) if self.edge_dim else layer(x, edge_index)
            x = self.leaky_relu(x)
            x = nn.functional.dropout(x, self.dropout, training=self.training)
        return x + x_res

    def forward(self, input_tensor, edge_index, edge_attr=None):
        x = input_tensor
        for index in range(len(self.ml)):
            # 训练时启用梯度检查点，节省显存
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self._forward_block, x, edge_index, index, edge_attr, use_reentrant=False)
            else:
                x = self._forward_block(x, edge_index, index, edge_attr)

        return x


class HyperGCN(nn.Module):
    """超图卷积网络，用于处理 KEGG 通路超图。

    结构和 GCN 几乎一样，只是把 ChunkedGCNConv 换成 HypergraphConv。
    超图卷积的消息传递是两阶段：节点 → 超边 → 节点。
    """
    def __init__(self,
                 in_features,
                 depths,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat=False,
                 dropout=0.5,
                 leaky_relu_slope=0.2,
                 gradient_checkpointing=False,
                 edge_chunk_size=262144,
    ):
        super(HyperGCN, self).__init__()
        self.ml = nn.ModuleList()
        self.res = nn.ModuleList()

        in_nodes = in_features
        for index in range(len(n_hidden)):
            res_layers = nn.ModuleList()
            nodes = n_hidden[index]
            feature_in = in_nodes
            for layer in range(depths[index]):
                # 唯一的区别：用 HypergraphConv
                layer_temp = HypergraphConv(in_channels=feature_in, out_channels=nodes)
                feature_in = nodes
                res_layers.append(layer_temp)

            if in_nodes == nodes:
                temp_res = nn.Identity()
            else:
                temp_res = nn.Linear(in_nodes, nodes)

            in_nodes = nodes
            self.ml.append(res_layers)
            self.res.append(temp_res)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing

    def _forward_block(self, x, edge_index, index):
        """一个模块的前向传播：若干层超图卷积 + 残差连接。"""
        x_res = self.res[index](x)
        for layer in self.ml[index]:
            x = layer(x, edge_index)
            x = self.leaky_relu(x)
            x = nn.functional.dropout(x, self.dropout, training=self.training)
        return x + x_res

    def forward(self, input_tensor, edge_index):
        x = input_tensor
        for index in range(len(self.ml)):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self._forward_block, x, edge_index, index, use_reentrant=False)
            else:
                x = self._forward_block(x, edge_index, index)

        return x


class DGHNN(nn.Module):
    """DGHNN 主模型：融合普通图和超图两个分支。

    流程：
      1. 普通图分支：GCN 处理 PPI 网络
      2. 元图分支：把普通图节点和元节点拼在一起，过一层 GCNConv
      3. 超图分支：HyperGCN 处理 KEGG 超图
      4. 拼接两个分支的输出，得到最终节点表示
    """
    def __init__(self,
                 in_features,
                 depths,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat=False,
                 dropout=0.5,
                 leaky_relu_slope=0.2,
                 meta_x=None,            # 元节点特征矩阵
                 data=None,              # PyG Data 对象
                 node2idx=None,          # (网络名, 基因名) -> 编号
                 gradient_checkpointing=False,
                 edge_chunk_size=262144,
    ):
        super(DGHNN, self).__init__()

        ppi_edge_attr = getattr(data, 'edge_attr', None)
        self.edge_dim = ppi_edge_attr.shape[1] if ppi_edge_attr is not None else 0
        # Static topology features are rebuilt from data/cache for checkpoint loading.
        # Exclude millions of edge descriptors from every best-model snapshot.
        self.register_buffer('ppi_edge_attr', ppi_edge_attr, persistent=False)

        # 普通图分支
        self.gat = GCN(in_features=in_features, depths=depths, n_hidden=n_hidden,
                       n_heads=n_heads, num_classes=num_classes, concat=concat,
                       dropout=dropout, leaky_relu_slope=leaky_relu_slope,
                       gradient_checkpointing=gradient_checkpointing,
                       edge_chunk_size=edge_chunk_size, edge_dim=self.edge_dim)

        # 超图分支
        self.hgat = HyperGCN(in_features=in_features, depths=depths, n_hidden=n_hidden,
                             n_heads=n_heads, num_classes=num_classes, concat=concat,
                             dropout=dropout, leaky_relu_slope=leaky_relu_slope,
                             gradient_checkpointing=gradient_checkpointing,
                             edge_chunk_size=edge_chunk_size)

        # 最终分类层（当前 forward 没用到，保留兼容）
        self.linear = nn.Linear(n_hidden[-1] * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout
        nn.init.xavier_normal_(self.linear.weight)

        # 构建元图：把每个网络的节点连接到对应的元节点
        x = data.x.float()
        self.nb_nodes = x.shape[0]

        # 把所有网络的节点名拼成一个大数组
        node_names = np.concatenate(data.node_names, axis=0)

        # meta_edge_index：普通图节点 → 元节点 的边
        # 第一行：普通图节点编号
        # 第二行：元节点编号（加偏移 x.shape[0]，避免和普通图节点冲突）
        meta_edge_index = [[], []]
        for i, node in enumerate(node_names):
            meta_edge_index[0].append(i)
            meta_edge_index[1].append(node2idx[tuple(node)] + x.shape[0])

        # 转成 tensor，加自环，注册为 buffer（随模型搬到 GPU）
        meta_edge_index = torch.tensor(meta_edge_index, dtype=torch.long)
        meta_edge_index, _ = add_self_loops(meta_edge_index)
        self.register_buffer("meta_edge_index", meta_edge_index)
        self.register_buffer("meta_x", torch.as_tensor(meta_x, dtype=torch.float32))

        # 元节点特征投影层
        self.meta_linear = nn.Linear(in_features, n_hidden[-1])

        # 元图对应边也从相同节点坐标构造 PE/SE，并实际参与传播。
        if self.edge_dim:
            from graph_encodings import build_edge_features
            if getattr(data, 'node_pe', None) is None or getattr(data, 'node_se', None) is None:
                raise ValueError('Encoded edges require node_pe and node_se')
            mapping = torch.tensor([node2idx[tuple(row)] for row in node_names], dtype=torch.long)
            # Every meta gene occurs in at least one original network.
            meta_pe = data.node_pe.new_zeros((len(meta_x), data.node_pe.shape[1]))
            meta_se = data.node_se.new_zeros((len(meta_x), data.node_se.shape[1]))
            meta_pe[mapping] = data.node_pe
            meta_se[mapping] = data.node_se
            meta_attr = build_edge_features(torch.cat((data.node_pe, meta_pe)),
                torch.cat((data.node_se, meta_se)), meta_edge_index)
            self.register_buffer('meta_edge_attr', meta_attr, persistent=False)
            self.meta_gnn = EdgeConditionedGCNConv(n_hidden[-1], n_hidden[-1],
                meta_attr.shape[1], edge_chunk_size=edge_chunk_size)
        else:
            self.register_buffer('meta_edge_attr', None, persistent=False)
            self.meta_gnn = GCNConv(n_hidden[-1], n_hidden[-1])

    def forward(self, input_tensor, adj_mat, h_matrix):
        # 1. 元节点特征投影
        meta_x = self.leaky_relu(self.meta_linear(self.meta_x))

        number_nodes_of_x = input_tensor.shape[0]
        number_nodes_of_meta_x = meta_x.shape[0]

        # 2. 普通图分支：GCN 处理 PPI 网络
        x = self.gat(input_tensor, adj_mat, self.ppi_edge_attr)

        # 3. 元图分支：把普通图节点和元节点拼在一起，过一层 GCNConv
        joint_x = torch.cat((x, meta_x), dim=0)
        if self.edge_dim:
            x = self.meta_gnn(joint_x, self.meta_edge_index, self.meta_edge_attr)
        else:
            x = self.meta_gnn(joint_x, self.meta_edge_index)
        x = self.leaky_relu(x)

        # 4. 拆分：保留统一基因元节点的部分
        _, x = torch.split(x, [number_nodes_of_x, number_nodes_of_meta_x], dim=0)

        # 5. 超图分支：HyperGCN 处理 KEGG 超图
        h_x = self.hgat(self.meta_x.to(dtype=input_tensor.dtype), h_matrix)

        # 6. 拼接两个分支的输出
        x = torch.cat((x, h_x), dim=1)

        return x

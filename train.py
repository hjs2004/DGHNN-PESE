import gcnIO
from torch_geometric.utils import add_self_loops
from utils import *
import os
import time
import json
import numpy as np
from model import DGHNN
from FT_model import FTC
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import itertools
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
from experiment_io import save_checkpoint, write_json
from graph_encodings import EncodingConfig, augment_dghnn_data

# 控制 CUDA 显存分配策略，减少显存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# ============================== 超参数 ==============================
# 优先用 CUDA，否则回退 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 是否启用混合精度训练（AMP）。只有 CUDA 才启用
AMP_ENABLED = DEVICE == "cuda"
AMP_DTYPE = torch.float16

# 输入数据类型：AMP 开启时用 float16，否则 float32
INPUT_DTYPE = AMP_DTYPE if AMP_ENABLED else torch.float32

LEARNING_RATE = 1e-4        # 学习率
BATCH_SIZE = 1000           # 分类器分批大小
NUM_EPOCH = int(os.environ.get("DGHNN_NUM_EPOCH", "3000"))  # 训练轮数，可通过环境变量覆盖
NUM_WORKERS = 1
PIN_MEMORY = False
LOAD_MODEL = True
# Linux 大显存卡使用实测更快的大分块；Windows/小显存卡保留原配置。
RUNTIME_PROFILE = os.environ.get("DGHNN_PROFILE", "auto")
if RUNTIME_PROFILE not in {"auto", "local", "server"}:
    raise ValueError("DGHNN_PROFILE must be auto, local, or server")
if RUNTIME_PROFILE == "auto":
    large_gpu = (DEVICE == "cuda" and os.name != "nt"
                 and torch.cuda.get_device_properties(0).total_memory >= 20 * 2**30)
    RUNTIME_PROFILE = "server" if large_gpu else "local"
EDGE_CHUNK_SIZE = int(os.environ.get(
    "DGHNN_EDGE_CHUNK_SIZE", "1048576" if RUNTIME_PROFILE == "server" else "32768"))
if EDGE_CHUNK_SIZE < 1:
    raise ValueError("DGHNN_EDGE_CHUNK_SIZE must be positive")
# 完整图关闭检查点在 RTX 3090 24GB 上仍 OOM，因此两种配置都保留检查点。
print(f"Runtime: profile={RUNTIME_PROFILE}, device={DEVICE}, AMP={AMP_ENABLED}, "
      f"edge_chunk_size={EDGE_CHUNK_SIZE}, classifier_batch={BATCH_SIZE}, "
      f"gradient_checkpointing={DEVICE == 'cuda'}", flush=True)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ENCODINGS_ENABLED = os.environ.get("DGHNN_ENCODINGS", "1")
if ENCODINGS_ENABLED not in {"0", "1"}:
    raise ValueError("DGHNN_ENCODINGS must be 0 or 1")
ENCODINGS_ENABLED = ENCODINGS_ENABLED == "1"

# 六个 HDF5 文件路径
paths = [
    str(DATA_DIR / "CPDB_multiomics.h5"),
    str(DATA_DIR / "IREF_multiomics.h5"),
    str(DATA_DIR / "IREF_2015_multiomics.h5"),
    str(DATA_DIR / "MULTINET_multiomics.h5"),
    str(DATA_DIR / "PCNET_multiomics.h5"),
    str(DATA_DIR / "STRINGdb_multiomics.h5"),
]

# 六个网络的名称
path_name = ["CPDB", "IREF", "IREF_2015", "MULTINET", "PCNET", "STRINGdb"]

# KEGG 通路文件和 NCBI 基因注释文件
pathway_file = str(PROJECT_ROOT / "KEGG_pathway_gene.txt")
mapping_file = str(DATA_DIR / "Homo_sapiens.gene_info")

# 留一网络法的轮数，默认 6
PATH_LIMIT = min(len(paths), int(os.environ.get("DGHNN_PATH_LIMIT", "6")))
PATH_INDICES = [int(i) for i in os.environ.get(
    "DGHNN_PATH_INDICES", ",".join(map(str, range(PATH_LIMIT)))).split(",") if i.strip()]
if not PATH_INDICES or len(set(PATH_INDICES)) != len(PATH_INDICES) or any(i not in range(len(paths)) for i in PATH_INDICES):
    raise ValueError("DGHNN_PATH_INDICES must contain unique network indices 0..5")

# 随机种子列表，默认 0~4
RANDOM_SEEDS = [int(seed) for seed in os.environ.get("DGHNN_SEEDS", "0,1,2,3,4").split(",") if seed.strip()]


def train(model, optimizer, criterion, classifier, features, adj_mat, h_matrix, target, index, scaler=None):
    """执行一轮训练。

    参数：
        model:        DGHNN 主模型
        optimizer:    优化器
        criterion:    损失函数（BCELoss）
        classifier:   FT-Transformer 分类器
        features:     所有节点的多组学特征
        adj_mat:      PPI 图的 edge_index
        h_matrix:     KEGG 超图的 h_edge_index
        target:       所有元节点的标签
        index:        当前训练集的元节点索引
        scaler:       AMP 梯度缩放器
    """
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    # 把数据搬到 DEVICE，并转成合适的 dtype
    features = torch.as_tensor(features, dtype=INPUT_DTYPE, device=DEVICE)
    adj_mat = torch.as_tensor(adj_mat, dtype=torch.int64, device=DEVICE)
    h_matrix = torch.as_tensor(h_matrix, dtype=torch.int64, device=DEVICE)
    target = torch.as_tensor(target, dtype=torch.float32, device=DEVICE)

    train_loss = 0
    auprc = 0
    auroc = 0

    # 清空梯度，切到训练模式
    optimizer.zero_grad(set_to_none=True)
    model.train()
    classifier.train()

    # 前向传播：DGHNN 提取节点表示（AMP 混合精度）
    with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
        output = model(features, adj_mat, h_matrix)

    # 只取训练节点的表示和标签
    output_, _target_ = output[index], target[index]
    n, dim = output_.size()

    # 分类器分批处理，降低显存占用
    training_times = int(np.ceil(n / BATCH_SIZE))
    output_chunks = []
    for inner_epoch in range(training_times):
        index_ = [i_temp for i_temp in range(inner_epoch * BATCH_SIZE, min(n, (inner_epoch + 1) * BATCH_SIZE))]
        features_, target_ = output_[index_], _target_[index_]

        # 分类器前向传播
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            output = classifier(features_)

        # 计算损失
        loss = criterion(output.float(), target_)
        train_loss += loss.item()

        # 最后一个批次不需要保留计算图
        retain_graph = inner_epoch < training_times - 1
        scaler.scale(loss).backward(retain_graph=retain_graph)
        output_chunks.append(output.detach().float().cpu())

    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    train_loss = train_loss / training_times

    # ============================== 计算训练指标 ==============================
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    output_all = torch.cat(output_chunks, dim=0)
    pred_y = output_all.numpy().tolist()
    ground_y = _target_.detach().float().cpu().numpy().tolist()

    # 按 0.5 阈值统计混淆矩阵
    for i in range(len(pred_y)):
        pred_o = pred_y[i][0]
        pred_two = 1 if pred_o > 0.5 else 0
        if ground_y[i][0] == 1:
            if pred_two == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_two == 1:
                FP += 1
            else:
                TN += 1

    # 计算 AUPRC
    try:
        temp_auprc = average_precision_score(ground_y, pred_y)
    except:
        temp_auprc = 0
    auprc += temp_auprc

    # 计算 AUROC
    try:
        temp_auroc = roc_auc_score(ground_y, pred_y)
    except:
        temp_auroc = 0
    auroc += temp_auroc

    # 计算其他指标
    accurancy = np.nan if (TP + FP + TN + FN) == 0 else (TP + TN) / (TP + FP + TN + FN)
    precision = np.nan if (TP + FP) == 0 else (TP) / (TP + FP)
    sensitivity = np.nan if (TP + FN) == 0 else (TP) / (TP + FN)
    specificity = np.nan if (FP + TN) == 0 else (TN) / (FP + TN)
    f1 = np.nan
    try:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    except:
        f1 = np.nan

    # 打印并写入日志
    print(f"Train Error: \n Avg loss: {train_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")
    result.write(f"Train Error: \n Avg loss: {train_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")


def test(model, criterion, classifier, features, adj_mat, h_matrix, target, index):
    """在验证集或测试集上评估模型，不更新参数。

    返回：
        test_loss, auprc, auroc
    """
    features = torch.as_tensor(features, dtype=INPUT_DTYPE, device=DEVICE)
    adj_mat = torch.as_tensor(adj_mat, dtype=torch.int64, device=DEVICE)
    h_matrix = torch.as_tensor(h_matrix, dtype=torch.int64, device=DEVICE)
    target = torch.as_tensor(target, dtype=torch.float32, device=DEVICE)

    test_loss = 0
    auprc = 0
    auroc = 0
    model.eval()
    classifier.eval()

    # 评估阶段不计算梯度
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            output = model(features, adj_mat, h_matrix)

        output_, _target_ = output[index], target[index]
        n, dim = output_.size()
        training_times = int(np.ceil(n / BATCH_SIZE))
        output_chunks = []
        for inner_epoch in range(training_times):
            index_ = [i_temp for i_temp in range(inner_epoch * BATCH_SIZE, min(n, (inner_epoch + 1) * BATCH_SIZE))]
            features_, target_ = output_[index_], _target_[index_]
            with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                output = classifier(features_)
            loss = criterion(output.float(), target_)
            test_loss += loss.item()
            output_chunks.append(output.detach().float().cpu())

        test_loss = test_loss / training_times

        # 指标计算同 train
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        output_all = torch.cat(output_chunks, dim=0)
        pred_y = output_all.numpy().tolist()
        ground_y = _target_.detach().float().cpu().numpy().tolist()
        for i in range(len(pred_y)):
            pred_o = pred_y[i][0]
            pred_two = 1 if pred_o > 0.5 else 0
            if ground_y[i][0] == 1:
                if pred_two == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_two == 1:
                    FP += 1
                else:
                    TN += 1

        try:
            temp_auprc = average_precision_score(ground_y, pred_y)
        except:
            temp_auprc = 0
        auprc += temp_auprc

        try:
            temp_auroc = roc_auc_score(ground_y, pred_y)
        except:
            temp_auroc = 0
        auroc += temp_auroc

        accurancy = np.nan if (TP + FP + TN + FN) == 0 else (TP + TN) / (TP + FP + TN + FN)
        precision = np.nan if (TP + FP) == 0 else (TP) / (TP + FP)
        sensitivity = np.nan if (TP + FN) == 0 else (TP) / (TP + FN)
        specificity = np.nan if (FP + TN) == 0 else (TN) / (FP + TN)
        f1 = np.nan
        try:
            f1 = 2 * precision * sensitivity / (precision + sensitivity)
        except:
            f1 = np.nan

        print(f"Test Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")
        result.write(f"Test Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")

    return test_loss, auprc, auroc


# ============================== 全局数据容器 ==============================
data_list = []              # 六个网络的 PyG Data 对象
node2idx = {}               # (网络名, 基因名) -> 统一元图编号
h_node2idx = {}             # 基因名 -> 统一元图编号
counter = 0                 # 元节点编号计数器
train_nodes = []            # 每个网络的训练节点名
test_nodes = []             # 每个网络的测试节点名
val_nodes = []              # 每个网络的验证节点名
meta_y = torch.zeros(1000000, 1)  # 元节点标签（预分配大数组）
y_list = []                 # 每个网络的合并标签
node_names_list = []        # 每个网络的节点名
number_nodes_each_graph = [] # 每个网络的节点数

# 固定特征顺序：4 类组学 × 16 个癌种 = 64 维
features_order = [
    'MF: UCEC', 'MF: BLCA', 'MF: THCA', 'MF: KIRC',
    'MF: READ', 'MF: LUAD', 'MF: ESCA', 'MF: LUSC',
    'MF: BRCA', 'MF: COAD', 'MF: HNSC', 'MF: KIRP',
    'MF: PRAD', 'MF: LIHC', 'MF: STAD', 'MF: CESC',
    'METH: UCEC', 'METH: BLCA', 'METH: THCA', 'METH: KIRC',
    'METH: READ', 'METH: LUAD', 'METH: ESCA', 'METH: LUSC',
    'METH: BRCA', 'METH: COAD', 'METH: HNSC', 'METH: KIRP',
    'METH: PRAD', 'METH: LIHC', 'METH: STAD', 'METH: CESC',
    'GE: UCEC', 'GE: BLCA', 'GE: THCA', 'GE: KIRC',
    'GE: READ', 'GE: LUAD', 'GE: ESCA', 'GE: LUSC',
    'GE: BRCA', 'GE: COAD', 'GE: HNSC', 'GE: KIRP',
    'GE: PRAD', 'GE: LIHC', 'GE: STAD', 'GE: CESC',
    'CNA: UCEC', 'CNA: BLCA', 'CNA: THCA', 'CNA: KIRC',
    'CNA: READ', 'CNA: LUAD', 'CNA: ESCA', 'CNA: LUSC',
    'CNA: BRCA', 'CNA: COAD', 'CNA: HNSC', 'CNA: KIRP',
    'CNA: PRAD', 'CNA: LIHC', 'CNA: STAD', 'CNA: CESC',
]

# 元节点特征矩阵
meta_x = torch.zeros((100000, 64))

# ============================== 读取并整理六个网络 ==============================
for path in paths:
    data = gcnIO.load_hdf_data(path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data

    # 解码特征名
    feature_names = [f.decode("utf-8") for f in feature_names]

    # 统一特征顺序
    feature_ind = [feature_names.index(f_n) for f_n in features_order]
    feature_names = np.array(feature_names)[feature_ind]
    features = features[:, feature_ind]
    number_nodes_each_graph.append(features.shape[0])

    # 构建统一节点编号
    for i in node_names:
        if (tuple(i) not in node2idx):
            node2idx[tuple(i)] = counter
            h_node2idx[i[1]] = counter
            counter += 1

    # 合并三个标签
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)
    y = torch.tensor(y_train).add(torch.tensor(y_test)).add(torch.tensor(y_val))
    y_list.append(y)

    # 写入元节点特征和标签
    for i, label in enumerate(y):
        idx = node2idx[tuple(node_names[i])]
        meta_x[idx] = torch.from_numpy(features[i])
        if (meta_y[idx] == 0):
            meta_y[idx] = label

    # 找出训练/验证/测试节点索引
    idx_train = [i for i, x in enumerate(train_mask) if x == 1]
    idx_val = [i for i, x in enumerate(val_mask) if x == 1]
    idx_test = [i for i, x in enumerate(test_mask) if x == 1]

    # 转成 Tensor
    adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(features)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 保存各划分的节点名
    train_nodes.append(node_names[idx_train])
    test_nodes.append(node_names[idx_test])
    val_nodes.append(node_names[idx_val])

    # 稠密邻接矩阵转 edge_index，加自环
    edge_index = (adj > 0).nonzero().t()
    if ENCODINGS_ENABLED:
        raw_edge_weight = adj[edge_index[0], edge_index[1]]
        edge_index, raw_edge_weight = add_self_loops(edge_index, raw_edge_weight, fill_value=1.)
    else:
        edge_index, _ = add_self_loops(edge_index)

    node_names_list.append(node_names)

    # 构建 PyG Data 对象
    data = Data(x=features, edge_index=edge_index, y=y, node_names=node_names)
    if ENCODINGS_ENABLED:
        data.edge_attr = raw_edge_weight[:, None]
    data_list.append(data)

# 截断元节点特征和标签
meta_x = meta_x[:len(node2idx)]
number_nodes_each_graph.append(meta_x.shape[0])

# 拼接所有网络的节点名
node_names = np.concatenate(node_names_list, axis=0)

# 构建元节点名称数组
meta_node_names = np.zeros(len(node2idx), dtype="object")
for k, v in node2idx.items():
    k1, k2 = k
    meta_node_names[v] = k2

# 元节点标签转 long
meta_y = meta_y[:len(node2idx)].to(dtype=torch.long)

# ============================== 构建 KEGG 超图 ==============================
gene_name_map = {}
gene_name_map_file = open(mapping_file, 'r').readlines()
for line in gene_name_map_file:
    line = line.split('\t')
    gene_name_map[line[1]] = line[2]

h_edge_index = []
count = 0
pathway = open(pathway_file, 'r').readlines()
for line in pathway:
    line = line.replace('\n', '').split('\t')
    for gene in line[2:]:
        if gene not in gene_name_map:
            continue
        gene_name = gene_name_map[gene]
        if gene_name not in h_node2idx.keys():
            continue
        a = h_node2idx[gene_name]
        h_edge_index.append([a, count])
    count += 1
h_edge_index = np.array(h_edge_index).T

# Topology-only PE/SE, once before the network/seed loops. One shared union graph
# gives every copy of a gene the same coordinate system (no cross-graph averaging).
encoding_metadata = {'enabled': False}
if ENCODINGS_ENABLED:
    encoding_config = EncodingConfig.from_env()
    print(f"Preparing node/edge PE+SE: {encoding_config}", flush=True)
    meta_x, encoding_metadata = augment_dghnn_data(
        data_list, meta_x, node2idx, encoding_config,
        cache_dir=os.environ.get('DGHNN_ENCODING_CACHE', str(PROJECT_ROOT / '.cache' / 'graph_encodings')),
    )
    print(f"PE/SE: node_dim={encoding_metadata['node_output_dim']}, "
          f"edge_dim={encoding_metadata['edge_output_dim']}, "
          f"RWSE={encoding_metadata['rw_method']}, cache_hit={encoding_metadata['cache_hit']}", flush=True)

# ============================== 训练结果目录 ==============================
default_output = 'DGHNN_PESE' if ENCODINGS_ENABLED else 'DGHNN'
save_dir = os.environ.get("DGHNN_OUTPUT_DIR", str(PROJECT_ROOT / 'outputs' / default_output))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
write_json(os.path.join(save_dir, 'encoding_config.json'), encoding_metadata)

# ============================== 留一网络测试 ==============================
for path_index in PATH_INDICES:
    data_name = path_name[path_index]
    save_dir_path = os.path.join(save_dir, str(data_name))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    random_seed = RANDOM_SEEDS
    auprc_ret = []
    auroc_ret = []
    for seed in random_seed:
        set_seed(seed)

        # 构造训练集：所有网络的训练节点 + 验证节点 + 非当前测试网络的测试节点
        train_set = {tuple(node) for t in train_nodes for node in t}
        t2 = {tuple(node) for t in val_nodes for node in t}
        t3 = set()
        for test_nodes_index in range(len(test_nodes)):
            if test_nodes_index == path_index:
                continue
            for node in test_nodes[test_nodes_index]:
                t3.add(tuple(node))
        train_set = train_set.union(t2)
        train_set = train_set.union(t3)

        # 从训练集里切 10% 做验证集
        val_set = set()
        for i, val in enumerate(itertools.islice(train_set, int(len(train_set) * 0.1))):
            val_set.add(val)

        # 当前网络的测试节点做测试集
        test_set = {tuple(node) for node in test_nodes[path_index]}

        # 防止数据泄漏
        train_set = train_set - train_set.intersection(test_set)
        val_set = val_set - val_set.intersection(test_set)
        train_set = train_set - val_set.intersection(val_set)

        # 转成元图索引
        idx_train = torch.tensor([node2idx[i] for i in train_set])
        idx_test = torch.tensor([node2idx[i] for i in test_set])
        idx_val = torch.tensor([node2idx[i] for i in val_set])
        split_record = {
            'network': data_name, 'seed': seed,
            'train_indices': idx_train.tolist(), 'validation_indices': idx_val.tolist(),
            'test_indices': idx_test.tolist(),
            'node_names': [str(name) for name in meta_node_names],
            'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        }
        write_json(os.path.join(save_dir_path, f'split{seed}.json'), split_record)

        # 把六个图拼成一个批次
        loader = DataLoader(data_list, batch_size=len(data_list))
        batch = next(iter(loader))
        features = batch.x
        edge_index = batch.edge_index

        # ============================== 初始化模型 ==============================
        model = DGHNN(
            in_features=features.shape[1],
            depths=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            n_hidden=[64, 64, 64, 64, 64, 64, 64, 64, 64],
            n_heads=0,
            num_classes=1,
            concat=False,
            dropout=0.1,
            leaky_relu_slope=0.2,
            meta_x=meta_x,
            data=batch,
            node2idx=node2idx,
            gradient_checkpointing=(DEVICE == "cuda"),
            edge_chunk_size=EDGE_CHUNK_SIZE,
        ).to(DEVICE)

        # FT-Transformer 分类器
        classifier = FTC(
            input_dim=128,
            num_classes=1,
            attn_drpout=0.1,
            ff_dropout=0.1,
            dim=16,
            depth=4,
            heads=4,
            dim_head=4,
        ).to(DEVICE)

        # 优化器
        optimizer = Adam([
            {'params': model.parameters()},
            {'params': classifier.parameters()},
        ], lr=LEARNING_RATE, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

        # 损失函数
        criterion = nn.BCELoss()

        # 日志文件
        result = open(os.path.join(save_dir_path, 'result' + str(seed) + '.txt'), 'w')

        best_loss = 99999
        best_epoch = 0
        best_auprc = 0
        best_auroc = 0
        save_dict = None

        # ============================== 训练循环 ==============================
        for epoch in range(NUM_EPOCH):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            epoch_started = time.perf_counter()
            print(f"Epoch: {epoch}\n")
            result.write(f"Epoch: {epoch}\n")
            train(model, optimizer, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_train, scaler)
            val_loss, val_auprc, val_auroc = test(model, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_val)
            test_loss, test_auprc, test_auroc = test(model, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_test)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_started
            peak_gib = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0
            timing = f"Epoch runtime: {elapsed:.2f}s, peak allocated GPU memory: {peak_gib:.2f} GiB\n"
            print(timing, flush=True)
            result.write(timing)
            result.flush()

            # 用验证集损失选最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_auprc = test_auprc
                best_auroc = test_auroc
                save_dict = save_checkpoint(
                    os.path.join(save_dir_path, f'best{seed}.pth'), model, classifier,
                    epoch=epoch, seed=seed, network=data_name,
                    validation_loss=val_loss, validation_auprc=val_auprc,
                    validation_auroc=val_auroc, test_auprc=test_auprc, test_auroc=test_auroc,
                    feature_order=features_order,
                    graph_encodings=encoding_metadata,
                )

            write_json(os.path.join(save_dir_path, f'metrics{seed}.json'), {
                'network': data_name, 'seed': seed, 'epoch': epoch,
                'epochs_completed': epoch + 1, 'epochs_requested': NUM_EPOCH,
                'complete': False, 'best_epoch': best_epoch,
                'best_validation_loss': best_loss,
                'test_auprc_at_best_validation': best_auprc,
                'test_auroc_at_best_validation': best_auroc,
                'epoch_seconds': elapsed, 'peak_allocated_gib': peak_gib,
                'graph_encodings': encoding_metadata,
            })

        print(f"Best Epoch: {best_epoch}\n")
        result.write(f"Best Epoch: {best_epoch}\n")

        # 保存最佳模型
        save_path = os.path.join(save_dir_path, 'min_loss' + str(seed) + '.pth')
        torch.save(save_dict['model_state'], save_path)

        # Mark complete only after all epochs and checkpoint serialization succeed.
        metrics_path = os.path.join(save_dir_path, f'metrics{seed}.json')
        with open(metrics_path, encoding='utf-8') as handle:
            final_metrics = json.load(handle)
        final_metrics['complete'] = True
        write_json(metrics_path, final_metrics)

        result.close()

        auprc_ret.append(best_auprc)
        auroc_ret.append(best_auroc)

    # ============================== 汇总五次实验结果 ==============================
    auprc_ret = np.array(auprc_ret)
    mean = auprc_ret.mean()
    std = auprc_ret.std()
    result_auprc = open(os.path.join(save_dir_path, 'result_auprc.txt'), 'w')
    result_auprc.write(str(round(mean, 3)) + '±' + str(round(std, 3)) + '\n')

    auroc_ret = np.array(auroc_ret)
    mean = auroc_ret.mean()
    std = auroc_ret.std()
    result_auprc.write(str(round(mean, 3)) + '±' + str(round(std, 3)) + '\n')

    result_auprc.close()

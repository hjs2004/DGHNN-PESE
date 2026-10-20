import h5py
import os
import numpy as np
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from datetime import datetime
from pathlib import Path


def load_hdf_data(path, network_name='network', feature_name='features'):
    """从 HDF5 容器中读取 GCN 训练所需的全部数据。

    这个函数读取一个已经预处理好的数据集，里面包含训练 GCN 模型
    所需的全部内容：网络结构、节点特征、节点名称、训练/验证/测试
    的划分和标签。

    参数：
    ---------
    path:                HDF5 文件路径
    network_name:        HDF5 里网络数据的键名。同一个文件里可能存了
                         多个网络，这个参数指定用哪一个。默认 'network'
    feature_name:        HDF5 里节点特征的键名。默认 'features'

    返回：
    一个元组，顺序为：
    network, features, y_train, y_val, y_test,
    train_mask, val_mask, test_mask, node_names, feature_names
    """
    with h5py.File(path, 'r') as f:
        # 读取邻接矩阵（网络结构）
        # 形状通常是 [num_nodes, num_nodes]，值表示节点之间是否有边
        network = f[network_name][:]

        # 读取节点特征矩阵
        # 形状通常是 [num_nodes, num_features]，每一行是一个基因的多组学特征
        features = f[feature_name][:]

        # 读取节点名称
        # 每一行是一个二元组，通常是 (网络名, 基因名)，例如 ('CPDB', 'TP53')
        node_names = f['gene_names'][:]

        # 读取训练集标签
        # 形状 [num_nodes, 1]，只标记训练集里的正类
        y_train = f['y_train'][:]

        # 读取测试集标签
        # 形状 [num_nodes, 1]，只标记测试集里的正类
        y_test = f['y_test'][:]

        # 读取验证集标签（可选）
        # 如果 HDF5 里没有 y_val，就设为 None
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None

        # 读取训练集掩码
        # 形状 [num_nodes]，1 表示该节点属于训练集，0 表示不属于
        train_mask = f['mask_train'][:]

        # 读取测试集掩码
        # 形状 [num_nodes]，1 表示该节点属于测试集
        test_mask = f['mask_test'][:]

        # 读取验证集掩码（可选）
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None

        # 读取特征名（可选）
        # 例如 ['MF: UCEC', 'METH: BLCA', ...]，用于对齐不同网络的特征顺序
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None

    # 把 node_names 里的 bytes 解码成字符串
    # 原始数据里 i[0] 和 i[1] 是 bytes，需要 decode 成 Python 字符串
    # 解码后每一行是 ['CPDB', 'TP53'] 这样的列表
    node_names = np.array([[i[0].decode("utf-8"), i[1].decode("utf-8")] for i in node_names])  # ours

    # 按固定顺序返回 10 个变量
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names


def write_hyper_params2(args, file_name):
    """把超参数写入磁盘（简化版）。

    把一组超参数写入文本文件。加载方式见 `load_hyper_params`。
    这个版本接受的是一个对象（比如 argparse.Namespace），
    通过 vars() 转成字典后遍历。

    参数：
    ----------
    args:               要保存的参数对象（通常是 argparse.Namespace）
    file_name:          输出文件名。为了让 load 函数正常工作，
                        通常命名为 'hyper_params.txt'
    """
    with open(file_name, 'w') as f:
        # vars(args) 把对象转成字典，然后逐行写入 "键\t值"
        for arg in vars(args):
            f.write('{}\t{}\n'.format(arg, getattr(args, arg)))
    print("Hyper-Parameters saved to {}".format(file_name))


def write_hyper_params(args, input_file, file_name):
    """把超参数写入磁盘（带 input_file 版本）。

    和 write_hyper_params2 类似，但 args 直接是字典，
    并且最后会额外写一行 input_file。

    参数：
    ----------
    args:               要保存的参数字典
    input_file:         输入数据文件路径，写到最后一行
    file_name:          输出文件名
    """
    with open(file_name, 'w') as f:
        # 遍历字典，写入 "键\t值"
        for arg in args:
            f.write('{}\t{}\n'.format(arg, args[arg]))
        # 最后写入 input_file，单独占一行
        f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))


def str_to_num(s):
    """把字符串转成整数或浮点数。

    尝试转 int，失败再尝试转 float，再失败就原样返回字符串。
    用于读取超参数文件时自动识别数值类型。
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_hyper_params(model_dir):
    """从磁盘读取超参数。

    读取之前用 `write_hyper_params` 保存的超参数文件。

    参数：
    ----------
    model_dir:          模型所在目录，里面应有 'hyper_params.txt'

    返回：
    一个字典，包含所有超参数；以及 input_file 路径。
    """
    file_name = os.path.join(model_dir, 'hyper_params.txt')
    input_file = None
    with open(file_name, 'r') as f:
        args = {}
        for line in f.readlines():
            if '\t' in line:
                # 有制表符的行是 "键\t值"
                key, value = line.split('\t')
                if value.startswith('['):
                    # 值以 [ 开头，说明是一个列表，比如 hidden dims
                    # 先把引号、空格、方括号都去掉，再按逗号分割
                    f = lambda x: "".join(c for c in x if c not in ['\"', '\'', ' ', '\n', '[', ']'])
                    l = [int(f(i)) for i in value.split(',')]
                    args[key.strip()] = l
                else:
                    # 普通值，用 str_to_num 自动识别类型
                    args[key.strip()] = str_to_num(value.strip())
            else:
                # 没有制表符的行是 input_file
                input_file = line.strip()
    # print ("Hyper-Parameters read from {}".format(file_name))
    return args, input_file


def create_model_dir(model=None):
    """创建一个带时间戳的模型保存目录。

    参数：
    ----------
    model:              可选，模型名。如果有，会加在目录名前缀

    返回：
    创建好的目录路径。
    """
    root_dir = './results/my_models/'
    # 如果根目录不存在，先创建
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
        print("Created Training Subdir")

    # 用当前时间生成时间戳，避免目录重名
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # 如果指定了模型名，就拼到时间戳前面
    if (model is not None):
        date_string = f'{model}_{date_string}'

    # 创建最终目录
    os.mkdir(os.path.join(root_dir, date_string))
    print("Created dir", os.path.join(root_dir, date_string))

    model_path = os.path.join(root_dir, date_string)
    return model_path


def save_predictions(output_dir, node_names, predictions):
    """把预测结果保存成 TSV 文件。

    参数：
    ----------
    output_dir:         输出目录
    node_names:         节点名称数组，每一行是 (网络名, 基因名)
    predictions:        预测概率，形状 [num_nodes, 2]
                        第 0 列是负类概率，第 1 列是正类概率
    """
    p = Path(output_dir)
    # 创建目录，parents=True 表示父目录不存在也一并创建
    p.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, 'predictions.tsv'), 'w') as f:
        # 写表头
        f.write('ID\tName\tProb_pos\n')
        # 逐行写：网络名、基因名、正类概率
        for pred_idx in range(predictions.shape[0]):
            f.write('{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                          node_names[pred_idx, 1],
                                          # predictions[pred_idx, 0])  ← 原注释：这是旧版写法
                                          predictions[pred_idx, 1])   # 取正类概率
                    )


def write_train_test_sets(out_dir, y_train, y_test, train_mask, test_mask):
    """把训练/测试集相关数据保存成 .npy 文件。

    参数：
    ----------
    out_dir:            输出目录
    y_train:            训练集标签
    y_test:             测试集标签
    train_mask:         训练集掩码
    test_mask:          测试集掩码
    """
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(out_dir, 'train_mask.npy'), train_mask)
    np.save(os.path.join(out_dir, 'test_mask.npy'), test_mask)


def read_train_test_sets(dir):
    """从 .npy 文件读取训练/测试集相关数据。

    参数：
    ----------
    dir:                数据所在目录

    返回：
    y_train, y_test, train_mask, test_mask
    """
    y_train = np.load(os.path.join(dir, 'y_train.npy'))
    y_test = np.load(os.path.join(dir, 'y_test.npy'))
    train_mask = np.load(os.path.join(dir, 'train_mask.npy'))
    test_mask = np.load(os.path.join(dir, 'test_mask.npy'))
    return y_train, y_test, train_mask, test_mask
# DGHNN 的节点/边 PE 与 SE

本次新增 GraphGPS 风格的结构预处理和边条件消息传递。参考用户提供论文的 Figure 1、Table 1，以及官方 `posenc_stats.py` 的 LapPE / RWSE 定义：

- 论文：<https://arxiv.org/abs/2205.12454>
- 官方统计代码：<https://github.com/rampasek/GraphGPS/blob/main/graphgps/transform/posenc_stats.py>
- 官方编码器：<https://github.com/rampasek/GraphGPS/tree/main/graphgps/encoder>

这是独立实现的 DGHNN 适配，不是完整 GraphGPS；未新增基因间全局注意力。原 FT-Transformer 仍是基因内部的特征 token 分类器。

## 编码定义与默认维度

| 对象 | 定义 | 默认维度 |
|---|---|---:|
| 节点原始特征 | 保留四组学×十六癌种全部64列 | 64 |
| 节点 PE | 对称归一化拉普拉斯的前8个非零模 | 8 |
| 节点 SE | 1、2、3、4、8步随机游走返回概率 | 5 |
| 节点增强输入 | 原始特征、PE、SE拼接 | **77** |
| 边原始特征 | 原邻接矩阵正边值；补充自环为1 | 1 |
| 边 PE | 两端 LapPE 的逐维绝对差 | 8 |
| 边 SE | 两端 RWSE 的均值及逐维绝对差 | 10 |
| 边增强输入 | 原始边值、相对PE、结构描述拼接 | **19** |

边指 PPI 普通边及网络节点到元节点的对应边。KEGG 超边的成员关系保持原样；其基因输入已同步扩展为77维，未将超边本身重新编码。模型隐藏层仍为64维，输出仍为128维，分类器接口不变。

## 共享坐标、稀疏计算和缓存

先按现有 `node2idx` 映射把六张 PPI 图投影到统一基因并图，编码拓扑使用无向、二值、去重、无自环的关联。同一基因的所有网络副本和元节点共享编码，避免分别求各图特征向量再错误地跨坐标平均。不会读取标签、划分或组学数值来计算编码。

该并图编码使用了六张 PPI 图的结构，符合当前模型的联合传导设置，不应据此声称未见网络归纳泛化。原有基因身份映射和划分问题未在本次特征扩展中修正。

LapPE 在每个连通分量独立计算；归一化前转float64，按排序明确排除每个分量的一个零模。孤立节点和不足的频率用0补齐。分量不超过256节点时精确分解；更大分量使用SciPy稀疏 `eigsh`，不构造大图的完整密集拉普拉斯矩阵。使用最大绝对值分量确定特征向量符号；这不是SignNet，不保证对重复特征值子空间旋转不变。

RWSE 在不超过256节点的图上精确计算 `diag(P^t)`；大图默认每节点256条带固定随机种子的游走估计，分批限制内存。因此真实大图的RWSE是近似值，增加walks可降低抽样误差，不能称其与官方小图密集实现逐值一致。

缓存位于 `.cache/graph_encodings/`，以二值拓扑、节点顺序、配置和实现版本共同校验。缓存只存节点PE/SE和元数据，边特征按原边顺序重建。更改标签不会影响编码；更改拓扑或参数会生成新缓存。多个进程可原子写入同一有效缓存；首次同时启动仍可能重复预计算，建议先单进程生成缓存后再排队。

## 边特征如何影响预测

新增 `EdgeConditionedGCNConv`：

```text
gate(u,v) = 2 * sigmoid(Linear(edge_attr(u,v)))
message(u,v) = fixed_GCN_normalization(u,v) * gate(u,v) * Linear(x_u)
```

每层有自己的可学习门控，初始权重/偏置为0，门控初始为1；随后通过任务损失学习边特征贡献。GCN拓扑度归一化保持固定，学习的是归一化消息的门控，并非重新计算加权度的标准加权GCN。边描述保持FP32，消息生成和聚合一起按块重计算，防止autograd保存所有边的 `[E, hidden]` 激活。原有块级梯度检查点也保留。

静态边特征注册为非持久buffer，随模型搬到GPU，但不重复写入每个checkpoint。加载增强模型时必须用相同数据/映射/编码配置先调用 `augment_dghnn_data` 重建Data和模型，再 `load_state_dict`；检查点的 `graph_encodings.topology_hash` 应与重建元数据一致。仅拷贝权重而没有相同图数据不够。

## 使用

`train.py` 默认启用PE/SE，首次运行自动预计算并缓存；默认输出目录为 `outputs/DGHNN_PESE`。已有启动脚本显式指定输出时，以其唯一目录为准。每次结果保存 `encoding_config.json`，最佳checkpoint及进度JSON也记录编码配置和拓扑哈希。

本地PowerShell（正式全图训练前确认硬件容量）：

```powershell
.\.venv\Scripts\python.exe -u train.py
```

只做真实CPDB子图的短检查，不读整个稠密邻接、不运行正式训练队列：

```powershell
.\.venv\Scripts\python.exe smoke_graph_encodings.py --nodes 1024 --epochs 2 --device cuda
```

关闭编码恢复原输入与原GCN路径，便于基线比较：

```powershell
$env:DGHNN_ENCODINGS = '0'
.\.venv\Scripts\python.exe -u train.py
```

可选环境变量：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `DGHNN_ENCODINGS` | `1` | `0`关闭；`1`启用 |
| `DGHNN_LAP_PE_DIM` | `8` | PE宽度，`0`仅使用SE |
| `DGHNN_RW_STEPS` | `1,2,3,4,8` | 正整数列表；空字符串仅使用PE |
| `DGHNN_RW_WALKS` | `256` | 大图每个节点的游走条数 |
| `DGHNN_ENCODING_SEED` | `0` | 固定预处理种子，与模型初始化种子分开 |
| `DGHNN_ENCODING_CACHE` | 项目根目录下 `.cache/graph_encodings` | 编码缓存位置 |

PE与SE不能同时关闭；完全恢复基线应设置 `DGHNN_ENCODINGS=0`。改变输入宽度后，旧64维模型不能作为新77维模型直接严格加载。

Linux可用 `DGHNN_ENCODINGS=0 /root/venvs/dghnn/bin/python -u train.py` 恢复基线。启用时默认 `python train.py` 即可；本轮未上传服务器、未启动3000轮或多卡正式训练。

## 验证范围

运行 `python -m unittest discover -s tests -v` 检查：精确RW返回概率、稀疏/采样路径、孤立节点、非零谱模、边顺序与无向对称、缓存命中/失效、真实特征保留、共享基因坐标、边门控作用及梯度、消息激活重计算，以及旧功能回归。

子图GPU冒烟和单元测试不证明全图显存足够，也不证明指标提升。首轮服务器全图任务应重新测峰值显存与时间，不能沿用旧无编码版本的耗时估计。

2026-10-20 本地验证：21项测试通过，其中10项覆盖本次PE/SE功能与回归；真实CPDB前1024节点子图含17958条PPI边（含自环）和8731个KEGG成员关联，完整27层主干与原分类器在CUDA AMP下完成2轮训练及评估。训练loss为0.60398、0.59371，边门控梯度有限，峰值分配显存约442.1 MiB，最终输出形状 `[1024, 1]`。这些loss仅是接线检查，不作为方法效果证据。

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat


class GEGLU(Module):
    """GEGLU 激活函数：x * GELU(x)

    一种门控激活，比 ReLU 更平滑，常用于 Transformer 的前馈层。
    """
    def forward(self, x):
        return x * F.gelu(x)


def FeedForward(dim, mult=4, dropout=0.):
    """Transformer 的前馈层。

    结构：LayerNorm → Linear → GEGLU → Dropout → Linear
    - dim: 输入维度
    - mult: 隐藏层扩展倍数，默认 4
    - dropout: dropout 概率
    """
    return nn.Sequential(
        nn.LayerNorm(dim),              # 层归一化
        nn.Linear(dim, dim * mult),     # 升维
        GEGLU(),                        # 门控激活
        nn.Dropout(dropout),            # dropout
        nn.Linear(dim * mult, dim)      # 降维回原维度
    )


class Attention(Module):
    """多头自注意力机制。

    - dim: 输入维度
    - heads: 注意力头数
    - dim_head: 每个头的维度
    - dropout: 注意力 dropout 概率
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads       # 所有头的总维度
        self.heads = heads
        self.scale = dim_head ** -0.5      # 缩放因子 1/sqrt(dim_head)

        self.norm = nn.LayerNorm(dim)      # 层归一化

        # QKV 投影：一次线性层生成 Q、K、V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # 输出投影
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        # 生成 Q、K、V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 重排成多头格式：[b, n, h*d] → [b, h, n, d]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        # 计算注意力分数：Q · K^T
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # softmax 归一化
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        # 加权求和：attn · V
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        # 重排回 [b, n, h*d]
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out, attn


class Transformer(Module):
    """Transformer 主体，由多个 Attention + FeedForward 组成。

    - dim: 特征维度
    - depth: Transformer 层数
    - heads: 注意力头数
    - dim_head: 每个头的维度
    - attn_dropout: 注意力 dropout
    - ff_dropout: 前馈层 dropout
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams=4
    ):
        super().__init__()

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
            ]))

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            # 自注意力
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            # 残差连接
            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(Module):
    """数值特征嵌入层。

    把每个数值特征投影成 dim 维向量。
    - dim: 嵌入维度
    - num_numerical_types: 数值特征的数量
    """
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        # 每个特征一个权重向量
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        # 每个特征一个偏置向量
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        # x: [b, n] → [b, n, 1]
        x = rearrange(x, 'b n -> b n 1')
        # 每个特征乘以自己的权重，加上偏置
        return x * self.weights + self.biases


class FTTransformer(Module):
    """FT-Transformer：用于表格数据的 Transformer 分类器。

    - input_dim: 输入数值特征数量
    - dim: 嵌入维度
    - depth: Transformer 层数
    - heads: 注意力头数
    - dim_head: 每个头的维度
    - dim_out: 输出维度（分类数）
    - attn_dropout: 注意力 dropout
    - ff_dropout: 前馈层 dropout
    """
    def __init__(
        self,
        input_dim,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        attn_dropout=0.,
        ff_dropout=0.,
    ):
        super().__init__()

        self.num_continuous = input_dim

        # 数值特征嵌入层
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token：一个可学习的向量，用于聚合全局信息
        self.cls_token = nn.Parameter(torch.ones(1, 1, dim))

        # Transformer 主体
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # 输出层：把 cls token 映射到分类输出
        self.to_logits = nn.Sequential(
            nn.Linear(dim, dim_out)
        )

        self.dropout = ff_dropout

    def forward(self, x_numer, return_attn=False):
        # 1. 数值特征嵌入
        x = self.numerical_embedder(x_numer)

        # 2. 加 cls token
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. dropout
        x = nn.functional.dropout(x, self.dropout, training=self.training)

        # 4. Transformer
        if return_attn:
            x, attns = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x, return_attn=False)

        # 5. 取 cls token 的输出
        x = x[:, 0]
        x = nn.functional.dropout(x, self.dropout, training=self.training)

        # 6. 分类输出
        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns
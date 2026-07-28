# 模型改进说明：V2/V3/V4 融合方式与 GAT-GAE 注意力机制

## 1. 文档目的

本文档对当前铜价区间预测模型中的两个核心改进进行说明：

```text
改进一：针对 V2、V3、V4 的融合方式，引入分支注意力融合机制。
改进二：针对 GAE 图编码器，引入 GAT 图注意力机制，构建 GAT-GAE。
```

这两个改进对应模型中的两个不同层面：

```text
V2/V3/V4 融合方式改进：
解决 market、demand、supply 三个图之间谁更重要的问题。

GAE 加入注意力机制：
解决每个图内部哪些邻居节点更重要的问题。
```

因此，两者并不是重复改进，而是分别作用在：

```text
跨图层面：三个图之间的融合
图内层面：单个图内部节点关系的聚合
```

## 2. 原始模型结构

当前模型的基础思路是：

```text
CNN 分支：
铜期货自身历史量价矩阵 X_t -> CNN -> V1

GAE 分支：
market 图 -> GAE -> V2
demand 图 -> GAE -> V3
supply 图 -> GAE -> V4

预测层：
融合 V1、V2、V3、V4 -> MLP -> 下一交易日铜价区间
```

其中：

```text
V1：铜期货自身历史量价特征
V2：宏观市场图特征
V3：需求端图特征
V4：供给端图特征
```

模型输出为：

```text
y_hat = [low_return_hat, high_return_hat]
```

即预测下一交易日铜价最低价收益率和最高价收益率。

## 3. 三版模型关系

当前项目中可以形成三版模型：

| 版本 | 图编码器 | V2/V3/V4 融合方式 | 作用 |
|---|---|---|---|
| 第一版 | GCN-GAE | 直接拼接 Concat | 基准模型 |
| 第二版 | GCN-GAE | 分支注意力融合 Branch Attention | 改进三个图之间的融合 |
| 第三版 | GAT-GAE | 分支注意力融合 Branch Attention | 同时改进图内聚合和跨图融合 |

对应代码配置为：

```text
第一版：
graph_encoder = "gcn"
fusion_mode = "concat"

第二版：
graph_encoder = "gcn"
fusion_mode = "branch_attention"

第三版：
graph_encoder = "gat"
fusion_mode = "branch_attention"
```

代码位置：

```text
src/copper_prediction/model.py
```

训练脚本位置：

```text
scripts/train.py
```

预测脚本位置：

```text
scripts/predict.py
```

---

# 第一部分：V2、V3、V4 融合方式的改进

## 4. 为什么要改进 V2/V3/V4 的融合方式

第一版模型中，三个 GAE 分支得到：

```text
V2 = GAE(G_market)
V3 = GAE(G_demand)
V4 = GAE(G_supply)
```

然后直接拼接：

```text
V_final = concat(V1, V2, V3, V4)
```

这种方法简单稳定，但存在一个问题：

```text
它没有显式判断 market、demand、supply 三个图在当前样本中的相对重要性。
```

铜价在不同市场阶段受到的主导因素不同。

例如：

```text
宏观金融冲击阶段：
DXY、利率、VIX、黄金、原油等 market 图信息可能更重要。

AI 与新能源需求强化阶段：
英伟达、台积电、特斯拉、比亚迪、电网设备等 demand 图信息可能更重要。

矿端供应扰动或库存快速变化阶段：
矿企股票、LME 库存、SHFE 库存、COMEX 库存等 supply 图信息可能更重要。
```

所以，模型不应在所有样本中都用固定方式处理三个图，而应该能够根据当前铜价状态动态调整三个图分支的贡献。

## 5. 原始拼接方式

第一版采用：

```text
V_fused = concat(V1, V2, V3, V4)
```

当前默认维度为：

```text
V1: 64 维
V2: 32 维
V3: 32 维
V4: 32 维
```

因此：

```text
V_fused: 64 + 32 + 32 + 32 = 160 维
```

然后：

```text
y_hat = MLP(V_fused)
```

该方式的优点：

```text
实现简单
参数较少
适合作为 baseline
训练相对稳定
```

该方式的不足：

```text
三个图分支没有显式权重
难以解释当前预测更依赖哪一类信息
无法动态刻画不同市场阶段的主导因素变化
```

## 6. 改进方案：分支注意力融合

第二版改进为：

```text
alpha_market, alpha_demand, alpha_supply = BranchAttention(V1, V2, V3, V4)

V_graph =
    alpha_market * V2
    + alpha_demand * V3
    + alpha_supply * V4

V_fused = concat(V1, V_graph)

y_hat = MLP(V_fused)
```

其中：

```text
alpha_market：market 图分支权重
alpha_demand：demand 图分支权重
alpha_supply：supply 图分支权重
```

并且：

```text
alpha_market + alpha_demand + alpha_supply = 1
```

这种设计的核心含义是：

```text
模型根据铜期货自身近期量价走势 V1，动态判断 market、demand、supply 三个图分支谁更重要。
```

## 7. 为什么用 V1 引导分支注意力

这里使用 `V1` 作为查询信号。

原因是：

```text
V1 来自铜期货自身历史量价矩阵，反映铜价自身的近期状态。
```

铜价自身状态可以帮助模型判断当前市场环境。

例如：

```text
如果铜价近期与风险资产同步大幅波动：
模型可能更关注 market 图。

如果铜价上行，同时需求端股票表现强：
模型可能更关注 demand 图。

如果铜价上涨，同时库存下降、矿企股票强：
模型可能更关注 supply 图。
```

因此，注意力融合不是静态地给三个图固定权重，而是：

```text
以 V1 为依据，对 V2、V3、V4 进行动态加权。
```

## 8. 分支注意力的数学表达

设：

```text
V1 ∈ R^64
V2 ∈ R^32
V3 ∈ R^32
V4 ∈ R^32
```

先将 V1 映射为查询向量：

```text
q = W_q V1
```

将三个图向量映射为键向量：

```text
k_market = W_k V2
k_demand = W_k V3
k_supply = W_k V4
```

将三个图向量映射为值向量：

```text
u_market = W_v V2
u_demand = W_v V3
u_supply = W_v V4
```

计算注意力分数：

```text
e_market = q^T k_market / sqrt(d)
e_demand = q^T k_demand / sqrt(d)
e_supply = q^T k_supply / sqrt(d)
```

其中：

```text
d：注意力隐藏维度，当前默认是 32
```

经过 softmax：

```text
[alpha_market, alpha_demand, alpha_supply]
= softmax([e_market, e_demand, e_supply])
```

得到融合图特征：

```text
V_graph =
    alpha_market * u_market
    + alpha_demand * u_demand
    + alpha_supply * u_supply
```

最终：

```text
V_fused = concat(V1, V_graph)
```

当前默认维度为：

```text
V1: 64 维
V_graph: 32 维
V_fused: 96 维
```

相比第一版：

```text
第一版 MLP 输入维度：160
第二版 MLP 输入维度：96
```

## 9. 分支注意力的代码实现

代码位置：

```text
src/copper_prediction/model.py
```

对应类：

```python
class BranchAttentionFusion(nn.Module):
```

该模块输入：

```text
v1:       [batch_size, 64]
v_market: [batch_size, 32]
v_demand: [batch_size, 32]
v_supply: [batch_size, 32]
```

该模块输出：

```text
fused:            [batch_size, 96]
branch_attention: [batch_size, 3]
v_graph:          [batch_size, 32]
```

其中：

```text
branch_attention[:, 0] = market_attention
branch_attention[:, 1] = demand_attention
branch_attention[:, 2] = supply_attention
```

模型配置项：

```python
fusion_mode: str = "concat"
branch_attention_dim: int = 32
```

当：

```text
fusion_mode = "concat"
```

模型使用第一版拼接方式。

当：

```text
fusion_mode = "branch_attention"
```

模型使用第二版分支注意力融合方式。

## 10. 分支注意力的解释性

预测时，如果模型使用：

```text
fusion_mode = "branch_attention"
```

则预测结果中可以输出：

```text
market_attention
demand_attention
supply_attention
```

例如：

```text
market_attention = 0.58
demand_attention = 0.27
supply_attention = 0.15
```

可以解释为：

```text
该交易日模型更依赖宏观市场图信息进行铜价区间预测。
```

如果某一段时间：

```text
demand_attention 持续上升
```

可以说明：

```text
模型在该阶段更关注需求端产业链信息。
```

如果：

```text
supply_attention 持续上升
```

可以结合矿端扰动、库存下降、仓单变化等供给端因素进行解释。

需要注意：

```text
注意力权重反映模型在预测任务中的关注程度，不等同于严格因果关系。
```

---

# 第二部分：GAE 部分加入注意力机制

## 11. 为什么要在 GAE 中加入注意力机制

当前 GAE 分支的基本形式为：

```text
Z = GCN(X, A)
A_hat = sigmoid(Z Z^T)
```

其中：

```text
X：节点特征矩阵
A：邻接矩阵
Z：节点嵌入矩阵
A_hat：重构邻接矩阵
```

GCN-GAE 的问题在于：

```text
节点聚合邻居信息时，主要依赖预先构造好的邻接矩阵。
```

邻接矩阵虽然来自相关系数，具有一定经济含义，但它仍然是人为构造的。

在铜价预测中，同一个图内部的不同邻居节点对铜价的重要性可能随时间变化。

例如 market 图中，铜节点可能连接：

```text
DXY
US 10Y Yield
Crude Oil
Gold
Nasdaq 100
VIX
```

不同阶段中，这些邻居的重要性并不相同。

因此，第三版希望让模型进一步学习：

```text
在同一个图内部，每个节点应该更关注哪些邻居节点。
```

这就是 GAE 中加入 GAT 注意力机制的动机。

## 12. GAT-GAE 的核心思想

GAT-GAE 的整体结构仍然是图自编码器：

```text
编码器：
Z = GAT(X, A)

解码器：
A_hat = sigmoid(Z Z^T)

训练目标：
让 A_hat 尽量接近 A
```

与原来的 GCN-GAE 相比：

```text
GCN-GAE：用 GCN 编码器得到 Z
GAT-GAE：用 GAT 编码器得到 Z
```

也就是说，第三版只改变：

```text
节点嵌入 Z 的编码方式
```

不改变：

```text
输入数据
邻接矩阵构造方式
内积解码方式
GAE 重构损失
最终区间预测目标
```

## 13. GAT 图注意力如何工作

对某个图：

```text
X ∈ R^(N x F)
A ∈ R^(N x N)
```

其中：

```text
N：节点数量
F：节点特征维度，当前为 8
```

批量训练时：

```text
X: [batch_size, N, F]
A: [batch_size, N, N]
```

### 13.1 节点线性变换

每个节点先进行线性变换：

```text
h_i = W x_i
```

如果：

```text
x_i ∈ R^8
```

则可以变为：

```text
h_i ∈ R^32
```

### 13.2 计算边注意力分数

对于有边的节点对 `(i, j)`，计算：

```text
e_ij = LeakyReLU(a^T [h_i || h_j])
```

其中：

```text
h_i：节点 i 的隐藏表示
h_j：节点 j 的隐藏表示
||：拼接
a：可学习参数
e_ij：节点 j 对节点 i 的原始注意力分数
```

### 13.3 邻接矩阵 mask

GAT 并不是让所有节点互相关注，而是在已有图结构约束下学习注意力。

如果：

```text
A_ij = 0
```

说明节点 i 和节点 j 没有边，则该位置被 mask：

```text
e_ij = -inf
```

softmax 后：

```text
alpha_ij = 0
```

也就是说：

```text
没有连边的节点不会参与邻居聚合。
```

如果：

```text
A_ij > 0
```

说明节点 i 和节点 j 有边，该边可以参与注意力计算。

当前代码还支持将邻接矩阵的边权作为先验加入 attention score：

```text
score_ij = attention_score_ij + log(A_ij)
```

这样做的含义是：

```text
相关系数构造出的边权仍然提供先验信息；
但最终邻居重要性还会由 GAT 学习得到。
```

### 13.4 softmax 得到注意力权重

对节点 i 的所有邻居做 softmax：

```text
alpha_ij =
exp(e_ij) / sum_{k in N(i)} exp(e_ik)
```

因此：

```text
sum_j alpha_ij = 1
```

### 13.5 加权聚合邻居

节点 i 的新表示为：

```text
z_i = sum_j alpha_ij h_j
```

例如：

```text
LME Copper 新表示
= 0.40 * DXY
+ 0.25 * Crude Oil
+ 0.15 * Gold
+ 0.10 * Nasdaq 100
+ 0.10 * VIX
```

这表示模型在 market 图内部认为这些邻居节点对铜节点的重要性不同。

## 14. 多头 GAT

当前代码中默认：

```text
gat_heads = 4
```

多头注意力的含义是：

```text
模型从多个子空间学习邻居关系。
```

可以理解为：

```text
第 1 个头可能更关注宏观金融联动
第 2 个头可能更关注商品联动
第 3 个头可能更关注风险资产联动
第 4 个头可能更关注短期波动联动
```

这些含义不是人为指定的，而是模型在训练中自动学习的。

当前 GAT-GAE 分支采用两层 GAT：

```text
第一层 GAT：
输入节点特征 8 维 -> 隐藏维度 32

第二层 GAT：
隐藏维度 32 -> 节点嵌入维度 16
```

为了保证维度一致：

```text
第一层使用多头 concat
第二层使用多头 average
```

最终输出：

```text
Z: [batch_size, num_nodes, 16]
```

## 15. GAT-GAE 的图向量构造

GAT 编码后得到：

```text
Z_market
Z_demand
Z_supply
```

每个图的节点嵌入维度仍然是：

```text
16
```

图向量仍然沿用原来的构造方式：

```text
graph_vector = concat(copper_embedding, graph_mean)
```

其中：

```text
copper_embedding：铜节点嵌入
graph_mean：全图所有节点嵌入的平均
```

所以：

```text
copper_embedding: 16 维
graph_mean: 16 维
graph_vector: 32 维
```

最终：

```text
V2: [batch_size, 32]
V3: [batch_size, 32]
V4: [batch_size, 32]
```

这使得 GAT-GAE 可以无缝接入第二版的分支注意力融合模块。

## 16. GAT-GAE 的代码实现

代码位置：

```text
src/copper_prediction/model.py
```

新增配置项：

```python
graph_encoder: str = "gcn"
gat_heads: int = 4
gat_attention_dropout: float = 0.1
gat_negative_slope: float = 0.2
gat_use_edge_weights: bool = True
```

其中：

```text
graph_encoder = "gcn"：使用原来的 GCN-GAE
graph_encoder = "gat"：使用新的 GAT-GAE
```

新增 GAT 层：

```python
class GraphAttentionLayer(nn.Module):
```

新增 GAT-GAE 分支：

```python
class GATGAEBranch(nn.Module):
```

主模型中通过配置选择图编码器：

```python
graph_branch = GAEBranch if self.config.graph_encoder == "gcn" else GATGAEBranch
```

然后三个图分支分别为：

```python
self.market_gae = graph_branch(...)
self.demand_gae = graph_branch(...)
self.supply_gae = graph_branch(...)
```

这样不需要重新写一个完整的模型类。

## 17. 为什么不单独重写一个新 model

项目中没有新建一个完整的：

```python
CopperIntervalPredictorV3
```

而是在原模型中加入可配置分支。

原因是：

```text
第一，CNN 分支不变。
第二，三个图输入不变。
第三，GAE 解码器不变。
第四，损失函数不变。
第五，训练脚本和预测脚本大部分逻辑不变。
```

如果重新写一个完整 model，会导致大量重复代码，并且不利于一、二、三版模型的公平对比。

现在的方式更适合实验：

```text
只改变 graph_encoder 和 fusion_mode 两个配置项；
其他训练条件尽量保持一致。
```

## 18. 损失函数是否改变

两个改进都不改变损失函数。

总损失仍然为：

```text
L_total = L_interval + lambda * L_reconstruction
```

其中：

```text
L_interval = MSE(y_hat, y) + bound_penalty
```

边界惩罚为：

```text
bound_penalty = mean(max(0, low_hat - high_hat)^2)
```

GAE 重构损失为：

```text
L_reconstruction =
1/3 * (
    MSE(A_market_hat, A_market)
    + MSE(A_demand_hat, A_demand)
    + MSE(A_supply_hat, A_supply)
)
```

完整写法为：

```text
L_total =
MSE(y_hat, y)
+ bound_penalty
+ lambda * 1/3 * (
    MSE(A_market_hat, A_market)
    + MSE(A_demand_hat, A_demand)
    + MSE(A_supply_hat, A_supply)
)
```

当前代码中默认：

```text
lambda = 0.1
```

对应训练参数：

```bash
--reconstruction-weight 0.1
```

## 19. 训练命令

### 19.1 第一版：GCN-GAE + Concat

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_v1_gcn_concat \
  --graph-encoder gcn \
  --fusion-mode concat \
  --epochs 50 \
  --batch-size 32
```

### 19.2 第二版：GCN-GAE + Branch Attention

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_v2_gcn_branch_attention \
  --graph-encoder gcn \
  --fusion-mode branch_attention \
  --branch-attention-dim 32 \
  --epochs 50 \
  --batch-size 32
```

### 19.3 第三版：GAT-GAE + Branch Attention

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_v3_gat_branch_attention \
  --graph-encoder gat \
  --gat-heads 4 \
  --gat-attention-dropout 0.1 \
  --fusion-mode branch_attention \
  --branch-attention-dim 32 \
  --epochs 50 \
  --batch-size 32
```

如果使用 manifest：

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --manifest data/manifest.csv \
  --output-dir outputs/models_v3_gat_branch_attention \
  --graph-encoder gat \
  --gat-heads 4 \
  --fusion-mode branch_attention \
  --epochs 50
```

## 20. 预测命令与输出解释

第二版或第三版预测命令：

```bash
.venv/bin/python scripts/predict.py \
  --raw-dir data/raw \
  --checkpoint outputs/models_v3_gat_branch_attention/best_model.pt \
  --date latest
```

如果使用的是 `fusion_mode = "branch_attention"` 的模型，预测结果会包含：

```text
market_attention
demand_attention
supply_attention
```

含义为：

```text
三个图分支在当前预测样本中的相对重要性。
```

如果使用的是 `graph_encoder = "gat"` 的模型，模型内部还会在 `aux` 中保留：

```text
market_gat_attention_1
market_gat_attention_2
demand_gat_attention_1
demand_gat_attention_2
supply_gat_attention_1
supply_gat_attention_2
```

这些张量的形状为：

```text
[batch_size, gat_heads, num_nodes, num_nodes]
```

含义为：

```text
每个图内部，不同节点之间的注意力权重。
```

当前 `predict.py` 默认只输出分支注意力权重。后续如果需要分析 GAT 内部节点注意力，可以单独写一个分析脚本，将铜节点对应的邻居注意力导出为 CSV。

## 21. 实验对比设计

建议使用三组模型进行对比：

| 模型 | 图编码器 | 融合方式 | 检验目的 |
|---|---|---|---|
| Model A | GCN-GAE | Concat | 基准模型 |
| Model B | GCN-GAE | Branch Attention | 检验 V2/V3/V4 融合改进是否有效 |
| Model C | GAT-GAE | Branch Attention | 检验 GAE 图内注意力是否进一步有效 |

控制变量：

```text
同一份数据
同样的训练集、验证集、测试集划分
同样的 CNN 窗口
同样的相关系数窗口
同样的节点特征
同样的预测目标
同样的 epoch
同样的 batch_size
同样的 learning_rate
同样的随机种子
```

比较指标：

```text
train_loss
val_loss
test_loss
MAE
RMSE
区间覆盖率
平均区间宽度
方向准确率
```

重点比较：

```text
Model B vs Model A：
判断分支注意力融合是否提升预测效果。

Model C vs Model B：
判断 GAT-GAE 图内注意力是否进一步提升预测效果。
```

## 22. 改进效果如何解释

### 22.1 如果第二版优于第一版

可以说明：

```text
market、demand、supply 三个图分支在不同样本中的贡献并不固定；
分支注意力融合能够更好地刻画不同市场阶段下三类信息的重要性变化。
```

论文中可以表述为：

```text
引入分支注意力机制后，模型能够根据铜期货自身历史量价特征，动态调整宏观市场、需求端与供给端图嵌入的权重，从而提高模型对不同市场状态的适应能力。
```

### 22.2 如果第三版优于第二版

可以说明：

```text
仅依赖预先构造的相关性邻接矩阵不足以充分刻画资产关系；
GAT-GAE 能够进一步学习图内部不同邻居节点的相对重要性；
图内部注意力有助于提升铜价区间预测能力。
```

论文中可以表述为：

```text
在 GAE 编码器中引入 GAT 注意力机制后，模型能够在给定资产关联网络的基础上，自适应学习不同邻居节点的信息贡献，从而更充分地刻画市场联动结构。
```

### 22.3 如果第三版没有优于第二版

可能原因包括：

```text
样本数量不足，GAT 参数更多，容易过拟合。
图结构本身较稀疏，GAT 可学习空间有限。
图结构过密，注意力分配不稳定。
节点特征噪声较高，导致图内注意力难以学出稳定模式。
```

可尝试：

```text
降低 gat_heads
提高 dropout
调整相关系数连边阈值
减少图节点数量
增加训练样本
增加 early stopping
```

## 23. 两个改进的关系总结

两个改进可以用一句话区分：

```text
V2/V3/V4 分支注意力：
判断三个图之间谁更重要。

GAT-GAE 图内注意力：
判断每个图内部哪些节点关系更重要。
```

更完整地说：

```text
第二版在图分支融合层面引入注意力机制，使模型能够动态调整 market、demand 和 supply 三类图信息的贡献权重。

第三版进一步在 GAE 编码器内部引入 GAT 注意力机制，使模型能够在每个图内部自适应学习不同邻居节点对目标节点的影响强度。
```

因此，最终第三版模型具有两层注意力：

```text
第一层：图内部注意力
例如 market 图中，铜节点更关注 DXY、原油、黄金还是 VIX。

第二层：跨图分支注意力
例如当前预测更依赖 market 图、demand 图还是 supply 图。
```

## 24. 最终模型结构

第三版最终结构为：

```text
CNN 分支：
X_t -> CNN1D -> V1

GAT-GAE 分支：
G_t^market -> GAT-GAE -> V2
G_t^demand -> GAT-GAE -> V3
G_t^supply -> GAT-GAE -> V4

分支注意力融合：
V1 引导 V2、V3、V4 的权重分配
V_graph = alpha_market * V2 + alpha_demand * V3 + alpha_supply * V4

预测层：
concat(V1, V_graph) -> MLP -> [low_return, high_return]
```

用公式概括：

```text
V1 = CNN(X_t)

V2 = GAT-GAE(G_t^market)
V3 = GAT-GAE(G_t^demand)
V4 = GAT-GAE(G_t^supply)

[alpha_market, alpha_demand, alpha_supply]
= BranchAttention(V1, V2, V3, V4)

V_graph =
    alpha_market * V2
    + alpha_demand * V3
    + alpha_supply * V4

y_hat = MLP(concat(V1, V_graph))
```

## 25. 可直接用于论文的方法描述

可以写成：

```text
为进一步提升模型对不同信息来源的自适应建模能力，本文从跨图融合和图内聚合两个层面对原始 CNN-GAE 模型进行改进。首先，在 market、demand 和 supply 三个 GAE 分支输出 V2、V3 和 V4 后，引入由 CNN 分支输出 V1 引导的分支注意力融合机制，使模型能够根据铜期货自身近期量价特征，动态调整宏观市场、需求端与供给端图嵌入的贡献权重。其次，在 GAE 编码器内部，将原有 GCN 编码器替换为 GAT 编码器，构建 GAT-GAE 分支，使模型在给定资产关联网络的基础上，进一步学习同一图内部不同邻居节点的信息贡献。最终，模型将 CNN 分支特征与注意力融合后的图特征输入 MLP 预测层，输出下一交易日铜价最低价收益率和最高价收益率。
```


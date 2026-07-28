# Branch Attention 与 GAT-GAE 具体实现说明

## 1. 文档说明

本文档根据当前模型代码，对两个核心改进进行详细说明：

```text
1. Branch Attention：用于改进 V2、V3、V4 的融合方式。
2. GAT-GAE：用于在 GAE 图编码器中加入图注意力机制。
```

这两个改进解决的是两个不同层面的问题：

```text
Branch Attention 解决的是：三个图分支之间谁更重要。
GAT-GAE 解决的是：每个图内部哪些邻居节点更重要。
```

因此，最终模型具有两层注意力：

```text
图间注意力：market、demand、supply 三个图之间的注意力。
图内注意力：单个图内部节点与邻居节点之间的注意力。
```

## 2. 整体模型背景

模型输入包括四类信息：

```text
X_cnn：铜期货自身历史量价矩阵
G_market：宏观市场图
G_demand：需求端图
G_supply：供给端图
```

模型先分别提取四个向量：

```text
V1 = CNN(X_cnn)
V2 = GraphEncoder(G_market)
V3 = GraphEncoder(G_demand)
V4 = GraphEncoder(G_supply)
```

其中：

```text
V1：铜自身历史量价特征
V2：宏观市场图特征
V3：需求端图特征
V4：供给端图特征
```

当前默认维度为：

```text
V1 ∈ R^64
V2 ∈ R^32
V3 ∈ R^32
V4 ∈ R^32
```

预测目标为：

```text
y_hat = [low_return_hat, high_return_hat]
```

即下一交易日铜价最低价收益率和最高价收益率。

---

# 第一部分：Branch Attention

## 3. 为什么需要 Branch Attention

原始融合方式是直接拼接：

```text
V_fused = concat(V1, V2, V3, V4)
```

该方式的输入维度为：

```text
64 + 32 + 32 + 32 = 160
```

然后将 `V_fused` 输入 MLP 预测层：

```text
y_hat = MLP(V_fused)
```

这种方法简单稳定，但存在一个问题：

```text
它没有显式建模 market、demand、supply 三个图分支在不同交易日的重要性变化。
```

铜价在不同市场阶段可能由不同因素主导：

```text
宏观冲击阶段：market 图可能更重要。
需求扩张阶段：demand 图可能更重要。
供给扰动阶段：supply 图可能更重要。
```

因此，模型需要一种机制，让它对每一个样本动态判断：

```text
当前更应该依赖 market 图、demand 图还是 supply 图。
```

Branch Attention 就是为了解决这个问题。

## 4. Branch Attention 的核心思想

Branch Attention 不再直接拼接：

```text
concat(V2, V3, V4)
```

而是先学习三个图分支的注意力权重：

```text
alpha_market
alpha_demand
alpha_supply
```

然后进行加权融合：

```text
V_graph =
    alpha_market * U_market
    + alpha_demand * U_demand
    + alpha_supply * U_supply
```

最后再与 `V1` 拼接：

```text
V_fused = concat(V1, V_graph)
```

其中：

```text
U_market、U_demand、U_supply 是 V2、V3、V4 经过 value 映射后的图向量。
```

最终预测：

```text
y_hat = MLP(V_fused)
```

## 5. 为什么用 V1 作为 Query

当前实现中，Branch Attention 使用 `V1` 作为 query。

原因是：

```text
V1 来自铜期货自身历史量价矩阵，反映铜价近期走势状态。
```

铜自身状态可以帮助模型判断当前阶段更像哪种市场环境：

```text
如果铜价近期与美元、利率、风险资产联动更强，market 图可能更重要。
如果铜价近期与 AI、新能源、电网设备相关资产联动更强，demand 图可能更重要。
如果铜价近期与矿企股票、库存变化联动更强，supply 图可能更重要。
```

所以实现逻辑是：

```text
以铜自身特征 V1 为查询信号，去询问 V2、V3、V4 三个图分支谁更重要。
```

## 6. Branch Attention 的输入输出

输入：

```text
V1       : [batch_size, 64]
V_market : [batch_size, 32]
V_demand : [batch_size, 32]
V_supply : [batch_size, 32]
```

其中：

```text
V_market = V2
V_demand = V3
V_supply = V4
```

首先将三个图向量堆叠：

```text
G = stack(V_market, V_demand, V_supply)
```

得到：

```text
G: [batch_size, 3, 32]
```

输出：

```text
V_fused          : [batch_size, 96]
branch_attention : [batch_size, 3]
V_graph          : [batch_size, 32]
```

其中：

```text
branch_attention[:, 0] = market_attention
branch_attention[:, 1] = demand_attention
branch_attention[:, 2] = supply_attention
```

## 7. Branch Attention 的具体计算过程

### 7.1 Query 映射

将 `V1` 映射为 query：

```text
q = W_q V1 + b_q
```

其中：

```text
V1 ∈ R^64
W_q ∈ R^(d_a x 64)
q ∈ R^d_a
```

当前默认：

```text
d_a = 32
```

因此：

```text
q ∈ R^32
```

批量形式为：

```text
Q: [batch_size, 1, 32]
```

这里多出的维度 `1` 表示只有一个 query，即铜自身特征 query。

### 7.2 Key 映射

将三个图向量映射为 key：

```text
k_market = W_k V_market + b_k
k_demand = W_k V_demand + b_k
k_supply = W_k V_supply + b_k
```

批量形式：

```text
K = W_k G + b_k
```

维度为：

```text
K: [batch_size, 3, 32]
```

三个 key 分别代表：

```text
market 图的匹配特征
demand 图的匹配特征
supply 图的匹配特征
```

### 7.3 Value 映射

将三个图向量映射为 value：

```text
u_market = W_v V_market + b_v
u_demand = W_v V_demand + b_v
u_supply = W_v V_supply + b_v
```

批量形式：

```text
U = W_v G + b_v
```

维度为：

```text
U: [batch_size, 3, 32]
```

当前实现中，value 映射后会经过 dropout：

```text
U_drop = Dropout(U)
```

这样可以降低融合层过拟合风险。

注意：当前实现没有对注意力权重本身做 dropout，而是对 value 做 dropout，因此输出的注意力权重仍然保持清晰可解释：

```text
alpha_market + alpha_demand + alpha_supply = 1
```

### 7.4 注意力分数

对 query 和每个 key 做点积：

```text
s_market = q^T k_market / sqrt(d_a)
s_demand = q^T k_demand / sqrt(d_a)
s_supply = q^T k_supply / sqrt(d_a)
```

写成统一形式：

```text
s_i = q^T k_i / sqrt(d_a), i ∈ {market, demand, supply}
```

除以：

```text
sqrt(d_a)
```

是为了避免注意力维度较大时点积值过大，导致 softmax 过于尖锐。

批量形式下：

```text
scores: [batch_size, 3]
```

### 7.5 softmax 得到分支权重

对三个分数做 softmax：

```text
alpha_i = exp(s_i) / sum_j exp(s_j)
```

其中：

```text
i, j ∈ {market, demand, supply}
```

得到：

```text
alpha = [alpha_market, alpha_demand, alpha_supply]
```

满足：

```text
alpha_market + alpha_demand + alpha_supply = 1
```

含义为：

```text
alpha_market：当前样本中 market 图的重要性
alpha_demand：当前样本中 demand 图的重要性
alpha_supply：当前样本中 supply 图的重要性
```

### 7.6 加权融合图向量

用注意力权重对 value 向量加权求和：

```text
V_graph =
    alpha_market * u_market
    + alpha_demand * u_demand
    + alpha_supply * u_supply
```

维度为：

```text
V_graph: [batch_size, 32]
```

### 7.7 最终融合向量

将 `V1` 与 `V_graph` 拼接：

```text
V_fused = concat(V1, V_graph)
```

维度为：

```text
V_fused: [batch_size, 64 + 32] = [batch_size, 96]
```

然后进入 MLP：

```text
y_hat = MLP(V_fused)
```

输出：

```text
y_hat: [batch_size, 2]
```

即：

```text
y_hat[:, 0] = low_return_hat
y_hat[:, 1] = high_return_hat
```

## 8. Branch Attention 的解释方式

假设某一天输出：

```text
market_attention = 0.62
demand_attention = 0.25
supply_attention = 0.13
```

可以解释为：

```text
该交易日模型更依赖宏观市场图信息进行预测。
```

如果某段时间：

```text
demand_attention 持续偏高
```

可以说明模型在该阶段更关注需求端产业链信息。

如果某段时间：

```text
supply_attention 持续偏高
```

可以结合矿企、库存、仓单、供给扰动等信息进行解释。

需要注意：

```text
Branch Attention 表示模型的预测关注权重，不等同于严格因果关系。
```

---

# 第二部分：GAT-GAE

## 9. 为什么需要 GAT-GAE

原始 GAE 分支使用 GCN 编码器：

```text
Z = GCN(X, A)
```

然后使用内积解码器重构邻接矩阵：

```text
A_hat = sigmoid(Z Z^T)
```

这种 GCN-GAE 的特点是：

```text
节点通过邻接矩阵 A 聚合邻居信息。
```

但是，GCN 的邻居聚合主要依赖预先构造好的邻接矩阵。虽然邻接矩阵来自相关系数，有一定经济含义，但它不能充分表达：

```text
在当前预测任务下，不同邻居节点对目标节点的重要性差异。
```

例如在 market 图中，铜节点可能同时连接：

```text
DXY
Crude Oil
Gold
VIX
Nasdaq 100
US 10Y Yield
```

不同阶段下，这些邻居对铜价预测的重要性可能不同。

GAT-GAE 的目标是：

```text
在 GAE 编码器内部加入图注意力机制，让模型学习每个节点应该更关注哪些邻居节点。
```

## 10. GAT-GAE 的整体结构

GAT-GAE 仍然是图自编码器：

```text
输入：节点特征 X 和邻接矩阵 A
编码：Z = GAT(X, A)
解码：A_hat = sigmoid(Z Z^T)
目标：让 A_hat 尽量接近 A
```

它与 GCN-GAE 的区别只在编码器：

```text
GCN-GAE：Z = GCN(X, A)
GAT-GAE：Z = GAT(X, A)
```

解码器仍然是：

```text
A_hat = sigmoid(Z Z^T)
```

重构损失仍然是：

```text
L_GAE = MSE(A_hat, A)
```

## 11. GAT-GAE 的输入维度

对于任意一个图：

```text
G = (X, A)
```

其中：

```text
X: [batch_size, N, F]
A: [batch_size, N, N]
```

含义为：

```text
batch_size：批量大小
N：图中的节点数量
F：节点属性维度
```

当前默认：

```text
F = 8
```

三个图的节点数不同：

```text
market 图：N = 20
demand 图：N = 18
supply 图：N = 20
```

但 GAT-GAE 的计算逻辑一致。

## 12. 单层 Graph Attention 的具体实现

当前 GAT 层是 dense multi-head GAT，即直接在小规模邻接矩阵上计算多头注意力。

设：

```text
X ∈ R^(B x N x F_in)
```

其中：

```text
B：batch_size
N：节点数
F_in：输入特征维度
```

### 12.1 多头线性变换

首先对节点特征做线性变换：

```text
H = X W
```

其中：

```text
W ∈ R^(F_in x (H_heads * F_out))
```

变换后 reshape 为：

```text
H ∈ R^(B x N x H_heads x F_out)
```

其中：

```text
H_heads：注意力头数量，当前默认 4
F_out：每个注意力头的输出维度
```

对于第 `r` 个注意力头：

```text
h_i^r ∈ R^F_out
```

表示节点 `i` 在第 `r` 个注意力头下的隐藏表示。

### 12.2 源节点和目标节点注意力打分

当前实现没有直接拼接 `[h_i || h_j]` 再乘一个大向量，而是使用等价的拆分形式：

```text
score_src_i^r = (a_src^r)^T h_i^r
score_dst_j^r = (a_dst^r)^T h_j^r
```

然后：

```text
e_ij^r = LeakyReLU(score_src_i^r + score_dst_j^r)
```

其中：

```text
a_src^r ∈ R^F_out
a_dst^r ∈ R^F_out
```

是第 `r` 个注意力头的可学习参数。

该形式等价于常见 GAT 中的：

```text
e_ij = LeakyReLU(a^T [h_i || h_j])
```

只是将 `a` 拆成了源节点部分和目标节点部分。

### 12.3 使用邻接矩阵作为 mask

当前 GAT 不是全连接注意力，而是在图结构约束下计算注意力。

定义边 mask：

```text
M_ij = 1, if A_ij > 0
M_ij = 0, if A_ij = 0
```

如果：

```text
M_ij = 0
```

则该位置不参与 softmax：

```text
e_ij^r = -inf
```

因此 softmax 后：

```text
alpha_ij^r = 0
```

这表示：

```text
无边节点不会参与邻居聚合。
```

### 12.4 使用边权作为注意力先验

当前实现中，如果启用边权先验，则会加入：

```text
log(A_ij)
```

即：

```text
e_ij^r = e_ij^r + log(A_ij)
```

这里要求：

```text
A_ij > 0
```

对于无边位置，仍然会被 mask 掉。

加入 `log(A_ij)` 的含义是：

```text
邻接矩阵中的相关系数强度仍然作为先验信息参与注意力计算。
```

如果某条边的权重较大，则在 softmax 之前获得更高先验分数；但最终注意力仍然由模型参数和节点特征共同决定。

因此，GAT-GAE 不是完全抛弃相关系数构图，而是：

```text
在相关系数图结构约束下，进一步学习节点间的动态重要性。
```

### 12.5 softmax 得到图内注意力

对每个节点 `i` 的所有邻居 `j` 做 softmax：

```text
alpha_ij^r =
exp(e_ij^r) / sum_{k ∈ N(i)} exp(e_ik^r)
```

其中：

```text
N(i)：节点 i 的邻居集合
r：第 r 个注意力头
```

满足：

```text
sum_{j ∈ N(i)} alpha_ij^r = 1
```

批量形式下，注意力张量维度为：

```text
Attention: [batch_size, heads, N, N]
```

其中：

```text
Attention[b, r, i, j]
```

表示第 `b` 个样本、第 `r` 个注意力头中，节点 `i` 对节点 `j` 的注意力权重。

### 12.6 attention dropout

训练时会对 attention 权重施加 dropout：

```text
alpha_drop = Dropout(alpha)
```

然后使用 dropout 后的注意力进行邻居聚合。

注意：

```text
模型保留下来的 attention 输出是 softmax 后、dropout 前的注意力。
```

这样更适合用于解释，因为它保持标准的归一化注意力分布。

### 12.7 加权聚合邻居

对每个注意力头：

```text
o_i^r = sum_{j ∈ N(i)} alpha_ij^r h_j^r
```

其中：

```text
o_i^r：节点 i 在第 r 个注意力头下的新表示
```

批量形式：

```text
O: [batch_size, heads, N, F_out]
```

## 13. 多头输出方式

当前实现支持两种多头输出方式：

```text
concat：将多个头拼接
average：将多个头平均
```

### 13.1 concat 输出

如果使用 concat：

```text
o_i = concat(o_i^1, o_i^2, ..., o_i^R)
```

输出维度：

```text
R * F_out
```

其中：

```text
R = heads
```

### 13.2 average 输出

如果使用 average：

```text
o_i = 1/R * sum_r o_i^r
```

输出维度：

```text
F_out
```

当前 GAT-GAE 使用：

```text
第一层 GAT：concat
第二层 GAT：average
```

这样既保留第一层多头表达能力，又保证第二层输出维度固定为节点嵌入维度。

## 14. 当前 GAT-GAE 两层结构

当前 GAT-GAE 编码器由两层 GAT 组成：

```text
GAT Layer 1
GAT Layer 2
```

默认参数：

```text
node_feature_dim = 8
gae_hidden_dim = 32
gae_embedding_dim = 16
gat_heads = 4
```

### 14.1 第一层 GAT

第一层输入：

```text
X: [batch_size, N, 8]
```

因为：

```text
gae_hidden_dim = 32
gat_heads = 4
```

所以每个头的输出维度为：

```text
hidden_per_head = 32 / 4 = 8
```

第一层每个头输出：

```text
[batch_size, N, 8]
```

四个头 concat 后：

```text
H1: [batch_size, N, 32]
```

然后经过：

```text
ELU 激活
Dropout
```

即：

```text
H1_drop = Dropout(ELU(H1))
```

### 14.2 第二层 GAT

第二层输入：

```text
H1_drop: [batch_size, N, 32]
```

第二层每个头输出：

```text
[batch_size, N, 16]
```

但第二层使用 average，而不是 concat。

因此最终输出：

```text
Z: [batch_size, N, 16]
```

这里的 `Z` 就是节点嵌入矩阵。

### 14.3 为什么第二层用 average

如果第二层也使用 concat，则输出维度会变成：

```text
gat_heads * gae_embedding_dim = 4 * 16 = 64
```

这会改变后续图向量维度。

为了保持与原 GAE 分支一致，第二层使用 average，使最终节点嵌入维度仍为：

```text
gae_embedding_dim = 16
```

这样 GAT-GAE 可以无缝替换 GCN-GAE。

## 15. GAT-GAE 的解码器

GAT 编码得到：

```text
Z ∈ R^(B x N x 16)
```

然后使用内积解码：

```text
A_hat = sigmoid(Z Z^T)
```

对单个样本：

```text
A_hat_ij = sigmoid(z_i^T z_j)
```

其中：

```text
z_i：节点 i 的嵌入向量
z_j：节点 j 的嵌入向量
```

批量形式：

```text
A_hat: [batch_size, N, N]
```

含义：

```text
A_hat_ij 表示模型根据节点嵌入重构出的节点 i 与节点 j 的连接强度。
```

## 16. GAT-GAE 的图向量构造

得到节点嵌入：

```text
Z: [batch_size, N, 16]
```

之后，每个图分支会构造一个 32 维图向量。

方法是：

```text
graph_vector = concat(copper_embedding, graph_mean)
```

其中：

```text
copper_embedding = Z 中铜节点对应的嵌入
graph_mean = 所有节点嵌入的平均
```

数学形式：

```text
z_copper = Z[:, copper_index, :]
```

```text
z_mean = 1/N * sum_i z_i
```

```text
V_graph_branch = concat(z_copper, z_mean)
```

维度：

```text
z_copper: [batch_size, 16]
z_mean: [batch_size, 16]
V_graph_branch: [batch_size, 32]
```

因此：

```text
V2 = concat(z_copper_market, z_mean_market)
V3 = concat(z_copper_demand, z_mean_demand)
V4 = concat(z_copper_supply, z_mean_supply)
```

这与原来的 GCN-GAE 分支输出维度完全一致。

## 17. 三个图分支的 GAT-GAE 输出

对于 market 图：

```text
X_market: [batch_size, 20, 8]
A_market: [batch_size, 20, 20]
Z_market: [batch_size, 20, 16]
V2: [batch_size, 32]
A_hat_market: [batch_size, 20, 20]
```

对于 demand 图：

```text
X_demand: [batch_size, 18, 8]
A_demand: [batch_size, 18, 18]
Z_demand: [batch_size, 18, 16]
V3: [batch_size, 32]
A_hat_demand: [batch_size, 18, 18]
```

对于 supply 图：

```text
X_supply: [batch_size, 20, 8]
A_supply: [batch_size, 20, 20]
Z_supply: [batch_size, 20, 16]
V4: [batch_size, 32]
A_hat_supply: [batch_size, 20, 20]
```

## 18. GAT-GAE 的图内注意力输出

GAT-GAE 会保留两层 GAT 的注意力权重：

```text
attention_1
attention_2
```

它们的形状为：

```text
[batch_size, gat_heads, N, N]
```

其中：

```text
attention_1：第一层 GAT 的图内注意力
attention_2：第二层 GAT 的图内注意力
```

对于 market 图：

```text
market_attention_1: [batch_size, 4, 20, 20]
market_attention_2: [batch_size, 4, 20, 20]
```

对于 demand 图：

```text
demand_attention_1: [batch_size, 4, 18, 18]
demand_attention_2: [batch_size, 4, 18, 18]
```

对于 supply 图：

```text
supply_attention_1: [batch_size, 4, 20, 20]
supply_attention_2: [batch_size, 4, 20, 20]
```

解释方式：

```text
attention[b, r, i, j]
```

表示：

```text
第 b 个样本中，第 r 个注意力头下，节点 i 对节点 j 的关注权重。
```

如果想分析铜节点关注哪些邻居，可以取：

```text
attention[b, r, copper_index, :]
```

这表示：

```text
铜节点对图中所有节点的注意力分布。
```

## 19. GAT-GAE 与 Branch Attention 的衔接

GAT-GAE 得到：

```text
V2: [batch_size, 32]
V3: [batch_size, 32]
V4: [batch_size, 32]
```

Branch Attention 使用：

```text
V1: [batch_size, 64]
V2: [batch_size, 32]
V3: [batch_size, 32]
V4: [batch_size, 32]
```

计算：

```text
V_graph =
    alpha_market * U_market
    + alpha_demand * U_demand
    + alpha_supply * U_supply
```

然后：

```text
V_fused = concat(V1, V_graph)
```

最终：

```text
y_hat = MLP(V_fused)
```

因此，完整第三版模型可以写成：

```text
V1 = CNN(X_cnn)

V2 = GAT-GAE(G_market)
V3 = GAT-GAE(G_demand)
V4 = GAT-GAE(G_supply)

[alpha_market, alpha_demand, alpha_supply]
= BranchAttention(V1, V2, V3, V4)

V_graph =
    alpha_market * U_market
    + alpha_demand * U_demand
    + alpha_supply * U_supply

y_hat = MLP(concat(V1, V_graph))
```

## 20. 损失函数

Branch Attention 和 GAT-GAE 都不改变损失函数。

总损失仍然为：

```text
L_total = L_interval + lambda * L_reconstruction
```

### 20.1 区间预测损失

模型预测：

```text
y_hat = [low_return_hat, high_return_hat]
```

真实标签：

```text
y = [low_return, high_return]
```

基础误差为：

```text
L_mse = MSE(y_hat, y)
```

为了避免模型预测的最低价收益率高于最高价收益率，加入边界惩罚：

```text
L_bound = mean(max(0, low_return_hat - high_return_hat)^2)
```

因此：

```text
L_interval = L_mse + beta * L_bound
```

当前默认：

```text
beta = 1.0
```

### 20.2 GAE 重构损失

每个图都有一个重构损失：

```text
L_market = MSE(A_hat_market, A_market)
```

```text
L_demand = MSE(A_hat_demand, A_demand)
```

```text
L_supply = MSE(A_hat_supply, A_supply)
```

三个图平均：

```text
L_reconstruction =
1/3 * (L_market + L_demand + L_supply)
```

### 20.3 总损失

最终：

```text
L_total =
L_interval + lambda * L_reconstruction
```

展开为：

```text
L_total =
MSE(y_hat, y)
+ beta * mean(max(0, low_return_hat - high_return_hat)^2)
+ lambda * 1/3 * (
    MSE(A_hat_market, A_market)
    + MSE(A_hat_demand, A_demand)
    + MSE(A_hat_supply, A_supply)
)
```

当前默认：

```text
lambda = 0.1
beta = 1.0
```

## 21. 两个注意力机制的区别

Branch Attention 的注意力对象是：

```text
market 图分支
demand 图分支
supply 图分支
```

它输出：

```text
[alpha_market, alpha_demand, alpha_supply]
```

回答的问题是：

```text
当前样本下，三个图谁更重要？
```

GAT-GAE 的注意力对象是：

```text
单个图内部的节点邻居关系
```

它输出：

```text
attention: [batch_size, heads, N, N]
```

回答的问题是：

```text
在某一个图内部，节点 i 更关注哪些邻居节点？
```

两者关系：

```text
GAT-GAE 先在每个图内部学习节点关系，得到 V2、V3、V4。
Branch Attention 再在三个图之间学习分支权重，得到融合图特征。
```

## 22. 最终改进模型总结

最终改进模型可以概括为：

```text
第一步：
用 CNN 从铜期货自身量价矩阵中提取 V1。

第二步：
用 GAT-GAE 分别从 market、demand、supply 三个图中提取 V2、V3、V4。

第三步：
用 V1 作为 query，对 V2、V3、V4 做 Branch Attention，得到三个图分支的权重。

第四步：
将三个图分支加权融合为 V_graph。

第五步：
将 V1 与 V_graph 拼接，输入 MLP，输出下一交易日铜价区间。
```

公式为：

```text
V1 = CNN(X_cnn)
```

```text
V2 = GAT-GAE(G_market)
V3 = GAT-GAE(G_demand)
V4 = GAT-GAE(G_supply)
```

```text
[alpha_market, alpha_demand, alpha_supply]
= softmax([
    q^T k_market / sqrt(d_a),
    q^T k_demand / sqrt(d_a),
    q^T k_supply / sqrt(d_a)
])
```

```text
V_graph =
    alpha_market * U_market
    + alpha_demand * U_demand
    + alpha_supply * U_supply
```

```text
y_hat = MLP(concat(V1, V_graph))
```

其中 GAT-GAE 内部为：

```text
Z = GAT(X, A)
```

```text
A_hat = sigmoid(Z Z^T)
```

```text
V_graph_branch = concat(z_copper, mean(Z))
```

## 23. 可用于论文的方法描述

可以写成：

```text
为提高模型对不同信息来源和资产关系结构的自适应建模能力，本文在原有 CNN-GAE 铜价区间预测模型基础上引入两类注意力机制。首先，在三个图分支输出 V2、V3 和 V4 后，设计由 CNN 分支输出 V1 引导的 Branch Attention 融合模块。该模块将 V1 映射为 query，将 market、demand 和 supply 三个图向量映射为 key 和 value，通过 scaled dot-product attention 计算三个图分支的权重，并对图向量进行加权融合，从而使模型能够根据铜期货自身近期量价状态动态调整宏观市场、需求端和供给端信息的贡献。

其次，在 GAE 分支内部，将原有 GCN 编码器替换为 GAT 编码器，构建 GAT-GAE。GAT 编码器在邻接矩阵约束下对节点邻居进行注意力加权聚合，并将邻接矩阵边权作为注意力分数的先验信息，使模型不仅利用相关性网络结构，还能够进一步学习同一图内部不同邻居节点的信息贡献。编码得到节点嵌入 Z 后，模型仍采用内积解码器 A_hat = sigmoid(ZZ^T) 重构邻接矩阵，并以重构误差作为 GAE 分支的辅助训练目标。最终，模型将 CNN 特征与 Branch Attention 融合后的图特征输入 MLP 预测层，输出下一交易日铜价最低价收益率和最高价收益率。
```


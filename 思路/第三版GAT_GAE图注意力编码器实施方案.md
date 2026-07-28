# 第三版模型实施方案：GAT-GAE 图注意力编码器

## 1. 第三版要解决的问题

第一版模型为：

```text
CNN + GCN-GAE + Concat
```

第二版模型为：

```text
CNN + GCN-GAE + Branch Attention
```

第二版已经解决了一个问题：

```text
market、demand、supply 三个图之间，当前样本下哪个图更重要？
```

但是第二版仍然保留了第一版中的 GCN-GAE 图编码器。也就是说，在每一个图内部，节点信息的聚合方式仍然主要由预先构造的邻接矩阵决定。

第三版要解决的问题是：

```text
在同一个图内部，不同邻居节点对目标节点的重要性是否应该由模型自适应学习？
```

例如在 `market` 图中，LME Copper 可能同时与以下节点存在连接：

```text
DXY
US 10Y Yield
Crude Oil
Gold
Nasdaq 100
S&P 500
VIX
```

在某些阶段，美元指数可能更重要；在另一些阶段，原油或风险资产可能更重要。普通 GCN-GAE 虽然会利用邻接矩阵，但它没有显式学习每个邻居节点的动态贡献权重。

因此，第三版的核心改动是：

```text
将 GAE 编码器中的 GCN 层替换为 GAT 层，形成 GAT-GAE。
```

一句话概括：

```text
第二版解决三个图之间谁更重要；第三版解决每个图内部哪些邻居节点更重要。
```

## 2. 第三版总体结构

第三版建议建立在第二版之上，即：

```text
CNN + GAT-GAE + Branch Attention
```

完整结构为：

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

对应三版模型关系：

```text
第一版：CNN + GCN-GAE + Concat
第二版：CNN + GCN-GAE + Branch Attention
第三版：CNN + GAT-GAE + Branch Attention
```

## 3. GCN-GAE 与 GAT-GAE 的区别

### 3.1 当前 GCN-GAE 的思想

当前代码中的 GAE 分支可以理解为：

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

GCN 的聚合方式可以简化理解为：

```text
节点 i 的新表示 = 邻居节点特征的加权平均
```

权重主要由邻接矩阵和度归一化决定。

如果两个节点在图中有边，GCN 会聚合它的信息；如果没有边，则不会直接聚合。

### 3.2 GAT-GAE 的思想

GAT-GAE 仍然保留 GAE 的整体框架：

```text
Z = GAT(X, A)
A_hat = sigmoid(Z Z^T)
```

不同点在于编码器由 GCN 换成 GAT。

GAT 会对每一条边学习一个注意力权重：

```text
alpha_ij = attention(node_i, node_j)
```

然后节点 i 的表示由邻居节点加权得到：

```text
z_i = sum_j alpha_ij W h_j
```

其中：

```text
alpha_ij：节点 j 对节点 i 的贡献权重
W h_j：节点 j 的线性变换表示
```

因此，GAT-GAE 不只是利用人工构建的相关性网络，还能进一步学习：

```text
在当前预测任务下，哪些邻居节点对铜价更有用。
```

## 4. 数学表达

### 4.1 输入

对任意一个图：

```text
G_t = (A_t, X_t)
```

其中：

```text
A_t ∈ R^(N x N)：第 t 日构造的邻接矩阵
X_t ∈ R^(N x F)：第 t 日节点属性矩阵
N：节点数量
F：节点属性维度，当前为 8
```

当前三个图为：

```text
G_t^market
G_t^demand
G_t^supply
```

### 4.2 节点特征线性变换

对每个节点特征做线性变换：

```text
h_i = W x_i
```

其中：

```text
x_i ∈ R^F
h_i ∈ R^D
```

`D` 是 GAT 的输出维度。

### 4.3 边注意力分数

对图中存在边的节点对 `(i, j)`，计算注意力分数：

```text
e_ij = LeakyReLU(a^T [h_i || h_j])
```

其中：

```text
||：向量拼接
a：可学习注意力参数
```

注意：只对邻接矩阵中存在连接的位置计算有效注意力。

如果：

```text
A_ij = 0
```

则节点 j 不参与节点 i 的邻居聚合。

### 4.4 注意力归一化

对节点 i 的所有邻居做 softmax：

```text
alpha_ij =
exp(e_ij) / sum_{k in N(i)} exp(e_ik)
```

其中：

```text
N(i)：节点 i 的邻居集合
```

这样可以保证：

```text
sum_j alpha_ij = 1
```

### 4.5 节点表示聚合

节点 i 的新表示为：

```text
z_i = sigma(sum_{j in N(i)} alpha_ij h_j)
```

其中：

```text
sigma：激活函数，例如 ELU 或 ReLU
```

### 4.6 GAE 解码

得到所有节点嵌入：

```text
Z_t = [z_1, z_2, ..., z_N]
```

然后仍然使用内积解码：

```text
A_hat_t = sigmoid(Z_t Z_t^T)
```

即：

```text
A_hat_ij,t = sigmoid(z_i,t^T z_j,t)
```

第三版不改变解码器，仍然沿用当前 GAE 的内积重构方式。

## 5. 第三版的图向量构造

当前 GAE 分支输出图向量的方式是：

```text
graph_vector = concat(copper_embedding, graph_mean)
```

其中：

```text
copper_embedding：铜节点的嵌入
graph_mean：全图节点嵌入的平均值
```

第三版建议继续沿用这个方式。

原因是：

```text
copper_embedding 代表目标资产铜在该图中的局部状态；
graph_mean 代表整个市场、需求端或供给端图的整体状态。
```

如果每个节点嵌入维度为 16，则：

```text
copper_embedding: 16 维
graph_mean: 16 维
graph_vector: 32 维
```

所以第三版仍然保持：

```text
V2 ∈ R^32
V3 ∈ R^32
V4 ∈ R^32
```

这样可以保证第二版和第三版在融合层维度上完全一致，便于公平对比。

## 6. 损失函数

第三版不改变损失函数。

仍然使用：

```text
L_total = L_interval + lambda * L_reconstruction
```

其中区间预测损失为：

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

完整形式：

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

第三版改变的是：

```text
Z_t 的编码方式：GCN -> GAT
```

不是改变：

```text
预测目标
损失函数
GAE 解码器
数据构造方式
```

## 7. 代码应该重新写一个 model，还是在原模型中加分支？

建议：

```text
不要重新写一个完整的新 model。
```

更推荐在原来的 `CopperIntervalPredictor` 中加入一个可配置的图编码器分支。

具体来说，保留统一主模型：

```python
class CopperIntervalPredictor(nn.Module):
```

新增配置项：

```python
graph_encoder: str = "gcn"
gat_heads: int = 4
gat_attention_dropout: float = 0.1
```

可选：

```text
graph_encoder = "gcn"：第一版、第二版使用当前 GCN-GAE
graph_encoder = "gat"：第三版使用 GAT-GAE
```

也就是说，最终通过两个配置项控制三版模型：

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

### 7.1 为什么不建议重新写完整 model

不建议新写一个完整模型类，例如：

```python
class CopperIntervalPredictorV3(nn.Module):
```

主要原因是：

```text
第一，重复代码会很多。
CNN 分支、三个图分支、GAE 解码、损失函数、训练脚本大部分都一样。

第二，对比实验不方便。
如果每一版都是一个新 model，训练脚本和 checkpoint 读取逻辑要不断分叉。

第三，容易出现非核心差异。
比如 V3 模型不小心改了 MLP 维度、dropout、初始化方式，就会影响实验公平性。

第四，维护成本更高。
后续如果修复损失函数或数据输入问题，需要同时改多个 model。
```

### 7.2 更推荐的代码组织方式

推荐采用：

```text
统一主模型 + 可替换图编码器模块
```

代码结构可以是：

```text
CopperIntervalPredictor
├── CopperCNN1D
├── market_gae: GAEBranch
├── demand_gae: GAEBranch
├── supply_gae: GAEBranch
├── BranchAttentionFusion
└── prediction head
```

第三版改为：

```text
CopperIntervalPredictor
├── CopperCNN1D
├── market_gae: GATGAEBranch
├── demand_gae: GATGAEBranch
├── supply_gae: GATGAEBranch
├── BranchAttentionFusion
└── prediction head
```

也就是三个图分支的类可替换，主模型接口不变。

训练脚本仍然调用：

```python
model = CopperIntervalPredictor(model_config)
```

只是配置不同：

```python
model_config = CopperIntervalPredictorConfig(
    graph_encoder="gat",
    fusion_mode="branch_attention",
)
```

## 8. 具体代码实施方案

### 8.1 配置项修改

在：

```text
src/copper_prediction/model.py
```

的 `CopperIntervalPredictorConfig` 中新增：

```python
graph_encoder: str = "gcn"
gat_heads: int = 4
gat_attention_dropout: float = 0.1
gat_negative_slope: float = 0.2
```

推荐默认仍然保持：

```text
graph_encoder = "gcn"
```

这样不会破坏第一版和第二版已有行为。

### 8.2 新增 GAT 层

新增一个图注意力层：

```python
class GraphAttentionLayer(nn.Module):
```

输入：

```text
x:         [batch_size, num_nodes, in_features]
adjacency: [batch_size, num_nodes, num_nodes]
```

输出：

```text
h_out:     [batch_size, num_nodes, out_features]
attention: [batch_size, heads, num_nodes, num_nodes]
```

核心步骤：

```text
1. 对节点特征做线性变换
2. 对每条边计算 attention score
3. 对非边位置 mask
4. 对每个节点的邻居维度做 softmax
5. 用 attention 加权聚合邻居节点
6. 多头输出 concat 或 average
```

### 8.3 新增 GAT-GAE 分支

新增：

```python
class GATGAEBranch(nn.Module):
```

结构建议：

```text
GAT layer 1:
node_feature_dim -> gae_hidden_dim
多头 concat

GAT layer 2:
gae_hidden_dim -> gae_embedding_dim
多头 average 或单头输出
```

为了让维度和当前模型保持一致，第二层最终输出必须是：

```text
[batch_size, num_nodes, gae_embedding_dim]
```

也就是默认：

```text
[batch_size, num_nodes, 16]
```

然后沿用原来的解码器：

```python
reconstruction = sigmoid(z @ z.transpose(1, 2))
```

以及原来的图向量：

```python
vector = concat(copper_embedding, graph_mean)
```

### 8.4 在主模型中选择分支

当前主模型里是：

```python
self.market_gae = GAEBranch(...)
self.demand_gae = GAEBranch(...)
self.supply_gae = GAEBranch(...)
```

第三版建议改为：

```python
branch_cls = GAEBranch if config.graph_encoder == "gcn" else GATGAEBranch

self.market_gae = branch_cls(...)
self.demand_gae = branch_cls(...)
self.supply_gae = branch_cls(...)
```

这样可以用同一个 `CopperIntervalPredictor` 实现三版模型。

### 8.5 训练脚本参数

在：

```text
scripts/train.py
```

新增参数：

```python
parser.add_argument("--graph-encoder", choices=["gcn", "gat"], default="gcn")
parser.add_argument("--gat-heads", type=int, default=4)
parser.add_argument("--gat-attention-dropout", type=float, default=0.1)
```

构建配置时：

```python
model_config = CopperIntervalPredictorConfig(
    graph_encoder=args.graph_encoder,
    fusion_mode=args.fusion_mode,
    branch_attention_dim=args.branch_attention_dim,
    gat_heads=args.gat_heads,
    gat_attention_dropout=args.gat_attention_dropout,
)
```

### 8.6 预测脚本

`scripts/predict.py` 不需要额外加命令行参数。

原因是 checkpoint 中已经保存了模型配置：

```python
"config": asdict(model.config)
```

预测时会自动读取：

```python
config = CopperIntervalPredictorConfig(**config_data)
```

所以只要训练时保存了：

```text
graph_encoder = "gat"
fusion_mode = "branch_attention"
```

预测时就能自动构建第三版模型。

如果希望进一步解释图内部注意力，可以让 `GATGAEBranch` 在 `aux` 中返回：

```text
market_node_attention
demand_node_attention
supply_node_attention
```

但第一步不建议过度展开输出，因为维度会比较大：

```text
[batch_size, heads, num_nodes, num_nodes]
```

可以先只实现模型训练和预测，后续再单独写分析脚本导出节点注意力。

## 9. 第三版训练命令

第一版：

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_concat \
  --graph-encoder gcn \
  --fusion-mode concat \
  --epochs 50 \
  --batch-size 32
```

第二版：

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_branch_attention \
  --graph-encoder gcn \
  --fusion-mode branch_attention \
  --branch-attention-dim 32 \
  --epochs 50 \
  --batch-size 32
```

第三版：

```bash
.venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models_gat_branch_attention \
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
  --output-dir outputs/models_gat_branch_attention \
  --graph-encoder gat \
  --gat-heads 4 \
  --fusion-mode branch_attention \
  --epochs 50
```

## 10. 第三版预测命令

```bash
.venv/bin/python scripts/predict.py \
  --raw-dir data/raw \
  --checkpoint outputs/models_gat_branch_attention/best_model.pt \
  --date latest
```

如果需要保存：

```bash
.venv/bin/python scripts/predict.py \
  --raw-dir data/raw \
  --checkpoint outputs/models_gat_branch_attention/best_model.pt \
  --date latest \
  --output outputs/predictions/gat_branch_attention_latest.csv
```

因为第三版建议继续使用第二版的分支注意力融合，所以预测结果仍然可以输出：

```text
market_attention
demand_attention
supply_attention
```

这三个权重解释的是：

```text
三个图之间谁更重要。
```

如果后续导出 GAT 内部注意力，则可以进一步解释：

```text
每个图内部哪些节点关系更重要。
```

## 11. 实验对比设计

第三版必须与第一版、第二版做对比。

建议实验表：

| 模型 | 图编码器 | 融合方式 | 目的 |
|---|---|---|---|
| Model A | GCN-GAE | Concat | 基准模型 |
| Model B | GCN-GAE | Branch Attention | 检验跨图分支注意力是否有效 |
| Model C | GAT-GAE | Branch Attention | 检验图内部注意力是否有效 |

为了公平比较，以下条件应保持一致：

```text
同一份数据
同样的训练集、验证集、测试集划分
同样的 CNN 窗口长度
同样的相关系数窗口
同样的节点特征
同样的目标变量
同样的 epoch、batch_size、learning_rate
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
Model B 是否优于 Model A：
说明跨图分支注意力是否有效。

Model C 是否优于 Model B：
说明图内部 GAT 注意力是否进一步有效。
```

## 12. 可能出现的问题

### 12.1 GAT 参数更多，可能更容易过拟合

GAT 比 GCN 参数更多，尤其是多头注意力会增加模型复杂度。

如果训练集较小，可能出现：

```text
train_loss 下降
val_loss 不下降甚至上升
```

解决方式：

```text
降低 gat_heads
增加 dropout
降低 gae_hidden_dim
增加 weight_decay
使用 early stopping
```

### 12.2 图太稠密时，GAT 注意力可能不稳定

如果邻接矩阵过密，每个节点邻居太多，GAT 需要在很多邻居之间分配注意力，可能导致学习困难。

解决方式：

```text
使用更严格的连边阈值
减少 Top-K
使用相关系数阈值 + 显著性检验
```

### 12.3 图太稀疏时，GAT 可学习空间有限

如果大部分节点只有自环，GAT 就退化成类似节点自身映射。

解决方式：

```text
降低相关系数阈值
保留自环
设置最少邻居数
采用阈值法与 Top-K 结合
```

### 12.4 GAT 内部注意力不等于因果关系

注意力权重可以帮助解释模型关注的信息，但不能直接等同于经济因果关系。

论文中应表述为：

```text
注意力权重反映模型在预测任务中对不同邻居节点信息的相对关注程度。
```

而不要写成：

```text
注意力权重证明某资产导致铜价变化。
```

## 13. 节点注意力如何解释

第三版如果导出 GAT 内部注意力，可以做两层解释：

第一层：跨图分支注意力。

```text
market_attention
demand_attention
supply_attention
```

解释：

```text
当前样本下，宏观市场、需求端、供给端谁更重要。
```

第二层：图内部节点注意力。

例如 market 图中铜节点对邻居的注意力：

```text
LME Copper -> DXY
LME Copper -> Crude Oil
LME Copper -> Gold
LME Copper -> Nasdaq 100
LME Copper -> VIX
```

解释：

```text
在 market 图内部，模型更关注哪些宏观金融资产与铜价的联动。
```

如果 demand 图中：

```text
LME Copper -> NVIDIA
LME Copper -> TSMC
LME Copper -> Tesla
LME Copper -> BYD
LME Copper -> NARI Technology
```

则可以解释为：

```text
模型在需求端图中更关注 AI 硬件、新能源车或电网设备链条。
```

如果 supply 图中：

```text
LME Copper -> Zijin Mining
LME Copper -> Freeport-McMoRan
LME Copper -> LME Copper Inventory
LME Copper -> SHFE Copper Inventory
```

则可以解释为：

```text
模型在供给端图中更关注矿企或库存变化。
```

## 14. 推荐实施顺序

不建议一上来同时改太多东西。

推荐顺序：

```text
第一步：保留当前第二版代码稳定运行
第二步：新增 graph_encoder 配置项
第三步：实现 GraphAttentionLayer
第四步：实现 GATGAEBranch
第五步：在 CopperIntervalPredictor 中按配置选择 GAEBranch 或 GATGAEBranch
第六步：给 train.py 增加 --graph-encoder、--gat-heads 等参数
第七步：跑 smoke_forward，确认 concat/gcn、branch_attention/gcn、branch_attention/gat 都能前向传播
第八步：用同一批数据跑三组模型对比
```

## 15. 第三版论文表述

可以写成：

```text
在第二版模型中，本文通过分支注意力机制实现了 market、demand 和 supply 三类图嵌入之间的自适应融合。然而，第二版模型中的图编码器仍采用 GCN 结构，其节点邻居信息聚合主要依赖预先构造的邻接矩阵，难以进一步刻画同一图内部不同邻居节点的重要性差异。

因此，第三版模型将 GAE 编码器中的 GCN 层替换为 GAT 层，构建 GAT-GAE 图编码分支。该结构在保留邻接矩阵约束的基础上，对每个节点的邻居节点学习注意力权重，使模型能够自适应识别不同资产关系在铜价区间预测中的相对贡献。最终，三个 GAT-GAE 分支分别输出 market、demand 和 supply 图嵌入，并通过分支注意力机制进行融合，再输入 MLP 预测层得到下一交易日铜价区间。
```

## 16. 第三版一句话总结

第三版可以概括为：

```text
在第二版分支注意力融合的基础上，将三个 GAE 分支内部的 GCN 编码器替换为 GAT 编码器，使模型不仅能判断 market、demand、supply 三个图谁更重要，还能进一步学习每个图内部不同邻居节点对铜价预测的相对重要性。
```


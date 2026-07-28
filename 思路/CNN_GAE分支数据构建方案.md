# 铜价区间预测模型的 CNN 分支与 GAE 分支数据构建方案

## 1. 研究目标

本文拟构建一个用于预测国际铜期货区间价格的深度学习模型。预测目标暂定为下一交易日 LME 铜的最低价和最高价：

$$
Y_t = [y^{low}_{t+1}, y^{high}_{t+1}]
$$

其中，为了降低价格量纲影响，建议将预测目标标准化为相对第 \(t\) 日收盘价的变化率：

$$
y^{low}_{t+1} = \frac{Low_{t+1} - Close_t}{Close_t}
$$

$$
y^{high}_{t+1} = \frac{High_{t+1} - Close_t}{Close_t}
$$

模型最终输出预测值后，可还原为真实价格：

$$
\widehat{Low}_{t+1} = Close_t \times (1 + \hat{y}^{low}_{t+1})
$$

$$
\widehat{High}_{t+1} = Close_t \times (1 + \hat{y}^{high}_{t+1})
$$

整体模型由两个主要部分组成：

1. CNN 分支：提取铜期货自身历史量价特征，输出特征向量 \(V_1\)。
2. GAE 分支：构建市场图、需求图和供给图，提取跨资产联动特征，分别输出 \(V_2\)、\(V_3\)、\(V_4\)。

最终将各分支特征拼接后输入 MLP，用于预测未来铜价区间：

$$
[\hat{y}^{low}_{t+1}, \hat{y}^{high}_{t+1}]
= MLP([V_1, V_2, V_3, V_4])
$$

## 2. CNN 分支：铜自身历史量价矩阵

### 2.1 分支作用

CNN 分支用于刻画铜期货自身的短期历史走势。它回答的问题是：

> 铜自己过去一段时间的价格、成交量、持仓量和波动形态如何变化？

该分支输入为第 \(t\) 日收盘后构造的铜自身历史量价矩阵：

$$
X_t^{CNN} \in R^{F \times L}
$$

其中：

- \(F\)：特征数量。
- \(L\)：历史窗口长度。
- 每一行表示一个铜自身特征。
- 每一列表示一个交易日。

第一版建议设置为：

$$
X_t^{CNN} \in R^{10 \times 20}
$$

即使用 10 个铜自身特征和过去 20 个交易日的数据。

### 2.2 原始数据字段

构建 CNN 分支至少需要 LME 铜的日频数据，包括：

| 字段 | 含义 |
|---|---|
| Open | 开盘价 |
| High | 最高价 |
| Low | 最低价 |
| Close | 收盘价 |
| Volume | 成交量 |
| OpenInterest | 持仓量 |

如果后续数据允许，可以进一步加入 Settlement、LME 库存、注销仓单、期限结构价差等变量。

### 2.3 每日特征构造

对每个交易日 \(s\)，构造 10 个铜自身特征：

$$
x_s^{CNN}
= [
open\_ret_s,
high\_ret_s,
low\_ret_s,
close\_ret_s,
intraday\_ret_s,
range_s,
upper\_shadow_s,
lower\_shadow_s,
volume\_change_s,
oi\_change_s
]
$$

具体定义如下：

| 编号 | 特征 | 公式 | 含义 |
|---|---|---|---|
| 1 | 开盘价相对变化 | \((Open_s - Close_{s-1}) / Close_{s-1}\) | 衡量开盘相对前收盘的跳空 |
| 2 | 最高价相对变化 | \((High_s - Close_{s-1}) / Close_{s-1}\) | 衡量当日上行空间 |
| 3 | 最低价相对变化 | \((Low_s - Close_{s-1}) / Close_{s-1}\) | 衡量当日下行空间 |
| 4 | 收盘价相对变化 | \((Close_s - Close_{s-1}) / Close_{s-1}\) | 日收益率 |
| 5 | 日内收益 | \((Close_s - Open_s) / Open_s\) | 当日多空方向 |
| 6 | 日内振幅 | \((High_s - Low_s) / Close_{s-1}\) | 当日波动幅度 |
| 7 | 上影线强度 | \((High_s - max(Open_s, Close_s)) / Close_{s-1}\) | 上方冲高回落程度 |
| 8 | 下影线强度 | \((min(Open_s, Close_s) - Low_s) / Close_{s-1}\) | 下方探底回升程度 |
| 9 | 成交量变化 | \(\log(Volume_s / Volume_{s-1})\) | 市场交易活跃度变化 |
| 10 | 持仓量变化 | \(\log(OpenInterest_s / OpenInterest_{s-1})\) | 资金参与和持仓变化 |

### 2.4 矩阵 \(X_t^{CNN}\) 的构造

在第 \(t\) 日收盘后，用过去 \(L=20\) 个交易日的数据构造矩阵：

$$
X_t^{CNN}
=
[x_{t-19}^{CNN}, x_{t-18}^{CNN}, ..., x_t^{CNN}]
$$

展开为：

$$
X_t^{CNN}
=
\begin{bmatrix}
open\_ret_{t-19} & open\_ret_{t-18} & ... & open\_ret_t \\
high\_ret_{t-19} & high\_ret_{t-18} & ... & high\_ret_t \\
low\_ret_{t-19} & low\_ret_{t-18} & ... & low\_ret_t \\
close\_ret_{t-19} & close\_ret_{t-18} & ... & close\_ret_t \\
intraday\_ret_{t-19} & intraday\_ret_{t-18} & ... & intraday\_ret_t \\
range_{t-19} & range_{t-18} & ... & range_t \\
upper\_shadow_{t-19} & upper\_shadow_{t-18} & ... & upper\_shadow_t \\
lower\_shadow_{t-19} & lower\_shadow_{t-18} & ... & lower\_shadow_t \\
volume\_change_{t-19} & volume\_change_{t-18} & ... & volume\_change_t \\
oi\_change_{t-19} & oi\_change_{t-18} & ... & oi\_change_t
\end{bmatrix}
$$

该矩阵作为 CNN 分支输入：

$$
V_1 = CNN(X_t^{CNN})
$$

### 2.5 CNN 输入形状

第一版明确采用一维 CNN。具体做法是：将 10 个铜自身特征视为 10 个输入通道，将过去 20 个交易日视为时间维度，并让卷积核沿时间维度滑动。

$$
X_t^{CNN}: [10, 20]
$$

batch 输入形状为：

$$
[batch\_size, 10, 20]
$$

其中：

- `batch_size`：样本数量。
- `10`：铜自身量价特征通道数。
- `20`：历史时间窗口长度。

一维 CNN 的卷积方向是时间维度，因此可以提取过去 20 个交易日中的 3 日、5 日或 7 日局部走势模式。例如，卷积核大小可以设置为 3、5 或 7，用于捕捉短期动量、波动扩大、冲高回落、探底回升等时序特征。

本文第一版不采用二维 CNN。原因是二维 CNN 会同时在“特征维度”和“时间维度”上卷积，而铜自身矩阵中的行是人为排列的特征，特征之间并不存在类似图像像素那样稳定的空间邻近关系。一维 CNN 将特征作为通道、只沿时间方向卷积，更符合时间序列预测任务。

## 3. GAE 分支：三类动态图网络

### 3.1 分支作用

GAE 分支用于刻画铜价所处的跨资产关系网络。CNN 分支关注铜自身历史走势，GAE 分支关注：

> 铜与宏观变量、商品、需求端资产和供给端资产之间最近如何联动？

每个预测时点 \(t\)，GAE 分支构建三个图：

$$
G_t^{market} = (A_t^{market}, X_t^{market})
$$

$$
G_t^{demand} = (A_t^{demand}, X_t^{demand})
$$

$$
G_t^{supply} = (A_t^{supply}, X_t^{supply})
$$

三个图分别表示：

| 图 | 含义 | 输出 |
|---|---|---|
| 市场图 \(G_t^{market}\) | 宏观与跨资产市场联动 | \(V_2\) |
| 需求图 \(G_t^{demand}\) | 铜需求端预期 | \(V_3\) |
| 供给图 \(G_t^{supply}\) | 铜供给端扰动、矿企和库存状态 | \(V_4\) |

### 3.2 GAE 输入的基本形式

每个图均由邻接矩阵和节点属性矩阵构成：

$$
G_t = (A_t, X_t)
$$

其中：

$$
A_t \in R^{N \times N}
$$

$$
X_t \in R^{N \times D}
$$

含义如下：

- \(N\)：图中节点数量，即资产或变量数量。
- \(D\)：每个节点的属性数量。
- \(A_t\)：节点之间的相关性网络。
- \(X_t\)：每个节点在第 \(t\) 日的状态特征。

第一版统一设置：

| 项目 | 建议 |
|---|----|
| 相关性窗口 | 60 个交易日 |
| 相关系数 | Spearman |
| 连边方式 | 每个节点保留 Top-5 绝对相关边 |
| 边权 | (corr_{ij,t}) |
| 节点属性数量 | 8  |
| GAE 嵌入维度 | 16 |

## 4. 市场图 \(G_t^{market}\)

### 4.1 图的作用

市场图用于刻画铜价所处的宏观金融环境和跨商品联动状态。它主要反映美元、利率、风险偏好、工业金属、能源和全球权益市场对铜价的共同影响。

### 4.2 建议节点

第一版市场图建议包含 20 个节点：

| 编号 | 节点 | 代表含义 |
|---|---|---|
| 1 | LME Copper | 目标资产 |
| 2 | COMEX Copper | 国际铜期货联动 |
| 3 | SHFE Copper | 中国铜期货联动 |
| 4 | Gold | 避险与美元定价商品 |
| 5 | Silver | 贵金属与工业属性 |
| 6 | Crude Oil | 能源成本与通胀 |
| 7 | Natural Gas | 能源价格 |
| 8 | Aluminum | 工业金属联动 |
| 9 | Zinc | 工业金属联动 |
| 10 | Nickel | 工业金属联动 |
| 11 | Lead | 工业金属联动 |
| 12 | Iron Ore | 黑色系与中国需求 |
| 13 | DXY | 美元指数 |
| 14 | US 10Y Yield | 美国利率环境 |
| 15 | VIX | 全球风险偏好 |
| 16 | S&P 500 | 全球权益风险资产 |
| 17 | Nasdaq 100 | 科技成长风险偏好 |
| 18 | MSCI Emerging Markets | 新兴市场需求预期 |
| 19 | CSI 300 | 中国宏观需求 |
| 20 | CNY/USD | 人民币汇率 |

因此：

$$
N_{market} = 20
$$

节点属性矩阵为：

$$
X_t^{market} \in R^{20 \times 8}
$$

邻接矩阵为：

$$
A_t^{market} \in R^{20 \times 20}
$$

## 5. 需求图 \(G_t^{demand}\)

### 5.1 图的作用

需求图用于刻画铜的消费预期。铜需求主要来自电力电网、新能源车、AI 数据中心、半导体、工业制造、基建和地产链等方向。

### 5.2 建议节点

第一版需求图建议先使用日频金融资产，便于实现：

| 编号 | 节点 | 代表含义 |
|---|---|---|
| 1 | LME Copper | 目标资产 |
| 2 | NVIDIA | AI 硬件与数据中心 |
| 3 | TSMC | 半导体制造 |
| 4 | AMD | AI 芯片 |
| 5 | Broadcom | AI 与通信芯片 |
| 6 | Eaton | 电力设备 |
| 7 | Schneider Electric | 电气设备 |
| 8 | ABB | 电气自动化 |
| 9 | Tesla | 新能源车 |
| 10 | BYD | 新能源车 |
| 11 | CATL / 宁德时代 | 动力电池 |
| 12 | NARI Technology / 国电南瑞 | 电网自动化 |
| 13 | China XD Electric / 中国西电 | 输变电设备 |
| 14 | Shanghai Electric / 上海电气 | 电力装备 |
| 15 | CSI New Energy Index | 中国新能源需求 |
| 16 | CSI Infrastructure Index | 中国基建需求 |
| 17 | CSI Real Estate Index | 地产链需求 |
| 18 | CSI 300 | 中国权益市场需求预期 |

因此：

$$
N_{demand} = 18
$$

节点属性矩阵为：

$$
X_t^{demand} \in R^{18 \times 8}
$$

邻接矩阵为：

$$
A_t^{demand} \in R^{18 \times 18}
$$

后续如果需要增强需求图，可以加入中国制造业 PMI、工业增加值、电网投资、地产销售面积等月频宏观变量。但第一版建议暂时不加入月频变量，以避免频率对齐带来的复杂性。

## 6. 供给图 \(G_t^{supply}\)

### 6.1 图的作用

供给图用于刻画铜矿供给、冶炼、库存、注销仓单、运输成本和能源成本等供给侧因素。

### 6.2 建议节点

第一版供给图建议包含 20 个节点：

| 编号 | 节点 | 代表含义 |
|---|---|---|
| 1 | LME Copper | 目标资产 |
| 2 | Freeport-McMoRan | 全球铜矿企 |
| 3 | Southern Copper | 全球铜矿企 |
| 4 | Antofagasta | 铜矿企 |
| 5 | First Quantum Minerals | 铜矿企 |
| 6 | Glencore | 大宗商品与铜矿 |
| 7 | BHP | 综合矿业 |
| 8 | Rio Tinto | 综合矿业 |
| 9 | Anglo American | 综合矿业 |
| 10 | Zijin Mining / 紫金矿业 | 中国矿企 |
| 11 | Jiangxi Copper / 江西铜业 | 铜冶炼与矿业 |
| 12 | Tongling Nonferrous / 铜陵有色 | 铜冶炼 |
| 13 | Yunnan Copper / 云南铜业 | 铜冶炼 |
| 14 | China Molybdenum / 洛阳钼业 | 有色矿业 |
| 15 | Crude Oil | 能源成本 |
| 16 | Baltic Dry Index | 运输成本与全球贸易 |
| 17 | LME Copper Inventory | LME 铜库存 |
| 18 | SHFE Copper Inventory | 上期所铜库存 |
| 19 | COMEX Copper Inventory | COMEX 铜库存 |
| 20 | LME Cancelled Warrants | LME 注销仓单 |

因此：

$$
N_{supply} = 20
$$

节点属性矩阵为：

$$
X_t^{supply} \in R^{20 \times 8}
$$

邻接矩阵为：

$$
A_t^{supply} \in R^{20 \times 20}
$$

## 7. GAE 节点属性构造

### 7.1 节点类型划分

GAE 三个图中的节点分为两类：

1. 金融资产类节点：包括期货、股票、指数、ETF、汇率、利率、波动率指数、运输指数等。这类节点通常具有价格序列，部分节点还具有 OHLC 和成交量数据。
2. 库存类节点：包括铜库存和注销仓单等。这类节点不是可交易金融资产，没有 OHLC 和成交量，因此需要使用库存变化类属性。

第一版节点类型划分如下。

**市场图 \(G_t^{market}\)**

市场图中的 20 个节点全部作为金融资产类节点处理：

| 节点编号 | 节点 | 节点类型 |
|---|---|---|
| 1 | LME Copper | 金融资产类 |
| 2 | COMEX Copper | 金融资产类 |
| 3 | SHFE Copper | 金融资产类 |
| 4 | Gold | 金融资产类 |
| 5 | Silver | 金融资产类 |
| 6 | Crude Oil | 金融资产类 |
| 7 | Natural Gas | 金融资产类 |
| 8 | Aluminum | 金融资产类 |
| 9 | Zinc | 金融资产类 |
| 10 | Nickel | 金融资产类 |
| 11 | Lead | 金融资产类 |
| 12 | Iron Ore | 金融资产类 |
| 13 | DXY | 金融资产类 |
| 14 | US 10Y Yield | 金融资产类 |
| 15 | VIX | 金融资产类 |
| 16 | S&P 500 | 金融资产类 |
| 17 | Nasdaq 100 | 金融资产类 |
| 18 | MSCI Emerging Markets | 金融资产类 |
| 19 | CSI 300 | 金融资产类 |
| 20 | CNY/USD | 金融资产类 |

其中，DXY、US 10Y Yield、VIX、CNY/USD 等节点虽然没有常规成交量，但仍属于金融资产类节点。构造属性时，`volume_change` 可设为 0。

**需求图 \(G_t^{demand}\)**

需求图中的 18 个节点全部作为金融资产类节点处理：

| 节点编号 | 节点 | 节点类型 |
|---|---|---|
| 1 | LME Copper | 金融资产类 |
| 2 | NVIDIA | 金融资产类 |
| 3 | TSMC | 金融资产类 |
| 4 | AMD | 金融资产类 |
| 5 | Broadcom | 金融资产类 |
| 6 | Eaton | 金融资产类 |
| 7 | Schneider Electric | 金融资产类 |
| 8 | ABB | 金融资产类 |
| 9 | Tesla | 金融资产类 |
| 10 | BYD | 金融资产类 |
| 11 | CATL / 宁德时代 | 金融资产类 |
| 12 | NARI Technology / 国电南瑞 | 金融资产类 |
| 13 | China XD Electric / 中国西电 | 金融资产类 |
| 14 | Shanghai Electric / 上海电气 | 金融资产类 |
| 15 | CSI New Energy Index | 金融资产类 |
| 16 | CSI Infrastructure Index | 金融资产类 |
| 17 | CSI Real Estate Index | 金融资产类 |
| 18 | CSI 300 | 金融资产类 |

**供给图 \(G_t^{supply}\)**

供给图同时包含金融资产类节点和库存类节点：

| 节点编号 | 节点 | 节点类型 |
|---|---|---|
| 1 | LME Copper | 金融资产类 |
| 2 | Freeport-McMoRan | 金融资产类 |
| 3 | Southern Copper | 金融资产类 |
| 4 | Antofagasta | 金融资产类 |
| 5 | First Quantum Minerals | 金融资产类 |
| 6 | Glencore | 金融资产类 |
| 7 | BHP | 金融资产类 |
| 8 | Rio Tinto | 金融资产类 |
| 9 | Anglo American | 金融资产类 |
| 10 | Zijin Mining / 紫金矿业 | 金融资产类 |
| 11 | Jiangxi Copper / 江西铜业 | 金融资产类 |
| 12 | Tongling Nonferrous / 铜陵有色 | 金融资产类 |
| 13 | Yunnan Copper / 云南铜业 | 金融资产类 |
| 14 | China Molybdenum / 洛阳钼业 | 金融资产类 |
| 15 | Crude Oil | 金融资产类 |
| 16 | Baltic Dry Index | 金融资产类 |
| 17 | LME Copper Inventory | 库存类 |
| 18 | SHFE Copper Inventory | 库存类 |
| 19 | COMEX Copper Inventory | 库存类 |
| 20 | LME Cancelled Warrants | 库存类 |

因此，供给图中：

- 1-16 号节点使用金融资产类节点属性。
- 17-20 号节点使用库存类节点属性。

虽然两类节点的经济含义不同，但都统一构造成 8 维节点属性，因此可以共同组成：

$$
X_t^{supply} \in R^{20 \times 8}
$$

### 7.2 金融资产节点属性

对于铜、商品、股票、指数、ETF 等日频金融资产，每个节点统一构造 8 个属性：

$$
x_{i,t}
=
[
r_{1d},
r_{5d},
r_{20d},
vol_{5d},
vol_{20d},
range,
volume\_change,
ma20\_gap
]
$$

具体定义如下：

| 编号 | 属性 | 公式 | 含义 |
|---|---|---|---|
| 1 | 1 日收益率 | \((Close_t - Close_{t-1}) / Close_{t-1}\) | 短期方向 |
| 2 | 5 日累计收益率 | \((Close_t - Close_{t-5}) / Close_{t-5}\) | 一周动量 |
| 3 | 20 日累计收益率 | \((Close_t - Close_{t-20}) / Close_{t-20}\) | 月度动量 |
| 4 | 5 日波动率 | \(std(r_{t-4:t})\) | 短期波动 |
| 5 | 20 日波动率 | \(std(r_{t-19:t})\) | 月度波动 |
| 6 | 日内振幅 | \((High_t - Low_t) / Close_{t-1}\) | 当日价格波动范围 |
| 7 | 成交量变化 | \(\log(Volume_t / Volume_{t-1})\) | 交易活跃度变化 |
| 8 | MA20 偏离 | \((Close_t - MA20_t) / MA20_t\) | 趋势偏离程度 |

对于没有成交量的变量，例如 DXY、VIX、利率、汇率等，可将 `volume_change` 设为 0，或者替换为一阶变化量。第一版建议统一设为 0，以保证处理简单。

### 7.3 库存类节点属性

对于 LME 铜库存、SHFE 铜库存、COMEX 铜库存和 LME 注销仓单等库存类变量，没有 OHLC 和成交量，因此使用另一组 8 维属性，但保持维度一致：

$$
x_{i,t}^{inventory}
=
[
chg_{1d},
chg_{5d},
chg_{20d},
vol_{5d},
vol_{20d},
rank_{252d},
ma20\_gap,
destock\_signal
]
$$

具体定义如下：

| 编号 | 属性 | 公式 | 含义 |
|---|---|---|---|
| 1 | 1 日变化率 | \((Inv_t - Inv_{t-1}) / Inv_{t-1}\) | 当日库存变化 |
| 2 | 5 日变化率 | \((Inv_t - Inv_{t-5}) / Inv_{t-5}\) | 短期累库或去库 |
| 3 | 20 日变化率 | \((Inv_t - Inv_{t-20}) / Inv_{t-20}\) | 月度库存趋势 |
| 4 | 5 日变化波动率 | \(std(chg_{t-4:t})\) | 库存短期波动 |
| 5 | 20 日变化波动率 | \(std(chg_{t-19:t})\) | 库存月度波动 |
| 6 | 252 日库存分位数 | 当前库存位于过去 252 日的分位数 | 库存高低位置 |
| 7 | MA20 偏离 | \((Inv_t - MA20_t) / MA20_t\) | 相对近期库存均值的偏离 |
| 8 | 去库/累库信号 | \(sign(chg_{5d})\) | 库存方向信号 |

## 8. GAE 邻接矩阵构造

### 8.1 收益率序列

对图中每个节点 \(i\)，计算其收益率或变化率序列：

$$
r_{i,t} = \frac{Close_{i,t} - Close_{i,t-1}}{Close_{i,t-1}}
$$

对于库存类节点，使用库存变化率：

$$
r_{i,t}^{inventory} = \frac{Inv_{i,t} - Inv_{i,t-1}}{Inv_{i,t-1}}
$$

### 8.2 相关性计算

在第 \(t\) 日，对任意两个节点 \(i\)、\(j\)，使用过去 60 个交易日的收益率序列计算 Spearman 相关系数：

$$
corr_{ij,t}
=
Spearman(
r_{i,t-59:t},
r_{j,t-59:t}
)
$$

得到原始相关矩阵：

$$
C_t \in R^{N \times N}
$$

### 8.3 Top-5 连边

为了避免图过密，对每个节点只保留相关性绝对值最高的 5 条边：

$$
A_{ij,t}
=
\begin{cases}
|corr_{ij,t}|, & j \in Top5_i(|corr_{ij,t}|) \\
0, & otherwise
\end{cases}
$$

其中，使用绝对值是因为负相关关系同样表示强联动。例如，美元指数与铜价可能为负相关，但这种关系对铜价预测仍然重要。

构造完成后，建议对邻接矩阵做对称化处理：

$$
A_t = \frac{A_t + A_t^T}{2}
$$

并加入自连接：

$$
A_t = A_t + I
$$

三个图分别构造自己的邻接矩阵：

$$
A_t^{market} \in R^{20 \times 20}
$$

$$
A_t^{demand} \in R^{18 \times 18}
$$

$$
A_t^{supply} \in R^{20 \times 20}
$$

## 9. GAE 编码与特征向量输出

### 9.1 GAE 编码

对每个图，将邻接矩阵和节点属性矩阵输入 GAE 编码器：

$$
Z_t^{market} = Encoder_{market}(A_t^{market}, X_t^{market})
$$

$$
Z_t^{demand} = Encoder_{demand}(A_t^{demand}, X_t^{demand})
$$

$$
Z_t^{supply} = Encoder_{supply}(A_t^{supply}, X_t^{supply})
$$

其中，若节点嵌入维度设为 16，则：

$$
Z_t^{market} \in R^{20 \times 16}
$$

$$
Z_t^{demand} \in R^{18 \times 16}
$$

$$
Z_t^{supply} \in R^{20 \times 16}
$$

### 9.2 GAE 训练目标

GAE 的训练目标是利用节点嵌入重构邻接矩阵。常见解码方式为内积解码：

$$
\hat{A}_{ij,t} = sigmoid(z_{i,t}^T z_{j,t})
$$

损失函数可以使用均方误差：

$$
L_{GAE} = ||A_t - \hat{A}_t||^2
$$

也可以使用二分类交叉熵：

$$
L_{GAE} = BCE(A_t, \hat{A}_t)
$$

通过重构资产关系网络，GAE 学到的节点嵌入 \(Z_t\) 能够反映当前市场联动结构。

### 9.3 图特征向量提取

GAE 输出的是每个节点的嵌入矩阵 \(Z_t\)，但后续 MLP 需要固定长度向量。因此，每个图提取两部分信息：

1. 铜节点嵌入：表示铜在当前网络中的位置。
2. 全图平均池化：表示整个图的总体状态。

市场图输出：

$$
V_2 = [z_{copper,t}^{market}, mean(Z_t^{market})]
$$

需求图输出：

$$
V_3 = [z_{copper,t}^{demand}, mean(Z_t^{demand})]
$$

供给图输出：

$$
V_4 = [z_{copper,t}^{supply}, mean(Z_t^{supply})]
$$

如果每个节点嵌入维度为 16，则：

$$
V_2, V_3, V_4 \in R^{32}
$$

## 10. 单个训练样本的最终结构

对每一个预测时点 \(t\)，一个训练样本包括：

| 部分 | 内容 | 形状 |
|---|---|---|
| CNN 输入 | 铜自身过去 20 日量价矩阵 \(X_t^{CNN}\) | \(10 \times 20\) |
| 市场图邻接矩阵 | \(A_t^{market}\) | \(20 \times 20\) |
| 市场图节点属性 | \(X_t^{market}\) | \(20 \times 8\) |
| 需求图邻接矩阵 | \(A_t^{demand}\) | \(18 \times 18\) |
| 需求图节点属性 | \(X_t^{demand}\) | \(18 \times 8\) |
| 供给图邻接矩阵 | \(A_t^{supply}\) | \(20 \times 20\) |
| 供给图节点属性 | \(X_t^{supply}\) | \(20 \times 8\) |
| 标签 | 下一日铜价最低价和最高价 | \(2\) |

即：

$$
Sample_t =
\{
X_t^{CNN},
A_t^{market}, X_t^{market},
A_t^{demand}, X_t^{demand},
A_t^{supply}, X_t^{supply},
Y_t
\}
$$

标签为：

$$
Y_t =
\left[
\frac{Low_{t+1} - Close_t}{Close_t},
\frac{High_{t+1} - Close_t}{Close_t}
\right]
$$

## 11. 模型融合结构

整体模型流程如下：

```text
铜自身历史量价矩阵 X_t^{CNN}
        -> CNN
        -> V1

市场图 G_t^{market}
        -> GAE_market
        -> V2

需求图 G_t^{demand}
        -> GAE_demand
        -> V3

供给图 G_t^{supply}
        -> GAE_supply
        -> V4

[V1, V2, V3, V4]
        -> MLP
        -> [预测最低价, 预测最高价]
```

公式表示为：

$$
V_1 = CNN(X_t^{CNN})
$$

$$
V_2 = GAE_{market}(G_t^{market})
$$

$$
V_3 = GAE_{demand}(G_t^{demand})
$$

$$
V_4 = GAE_{supply}(G_t^{supply})
$$

$$
\hat{Y}_t = MLP([V_1, V_2, V_3, V_4])
$$

## 12. 数据实现注意事项

### 12.1 时间顺序

第 \(t\) 日样本只能使用第 \(t\) 日及以前的数据。预测标签才是第 \(t+1\) 日的最低价和最高价。尤其在构造邻接矩阵时，只能使用：

```text
t-59 到 t 的收益率
```

不能使用第 \(t+1\) 日的数据，否则会发生数据泄露。

### 12.2 交易日对齐

不同市场交易日不同，例如 LME、美股、A 股、欧洲股票和中国期货交易日并不完全一致。第一版建议：

1. 建立统一日期索引。
2. 对价格类数据进行 forward-fill。
3. 对收益率缺失值设为 0。
4. 对成交量缺失值设为 0 或 forward-fill 后再计算变化率。

### 12.3 节点集合固定

第一版建议固定三个图的节点集合，不要每天动态更换节点。动态筛选 Top-m 资产可以作为后续扩展，否则会增加实现难度，也会削弱模型解释性。

### 12.4 月频宏观变量处理

PMI、工业增加值、电网投资等月频变量有经济含义，但频率较低。第一版建议暂时不放入 GAE 图中，可以后续作为外生控制变量加入 MLP，或者通过 forward-fill 转换为日频后再加入。

### 12.5 第一版推荐配置

| 模块 | 推荐设置 |
|---|---|
| 预测目标 | 下一日 LME 铜最低价和最高价 |
| CNN 类型 | 一维 CNN |
| CNN 窗口 | 20 个交易日 |
| CNN 特征数 | 10 |
| 市场图节点数 | 20 |
| 需求图节点数 | 18 |
| 供给图节点数 | 20 |
| GAE 节点属性数 | 8 |
| 相关性窗口 | 60 个交易日 |
| 连边方式 | Top-5 绝对相关边 |
| GAE 嵌入维度 | 16 |
| GAE 输出 | 铜节点嵌入 + 全图平均池化 |

## 13. 可用于论文写作的表述

本文构建了融合铜自身量价特征与跨资产图结构特征的铜价区间预测模型。对于 CNN 分支，本文以 LME 铜过去 20 个交易日的开盘价、最高价、最低价、收盘价、日内收益、振幅、影线、成交量变化和持仓量变化等 10 个特征构建历史量价矩阵，并采用一维 CNN 沿时间维度提取铜价自身的短期时序模式。

对于 GAE 分支，本文将铜价相关资产划分为市场端、需求端和供给端三个子系统，并分别构建动态图网络。市场图包含铜期货、主要工业金属、能源、美元指数、利率、波动率和主要权益指数，用于刻画宏观与跨资产联动；需求图包含 AI 硬件、新能源车、电力设备、基建和地产链相关资产，用于刻画铜消费预期；供给图包含全球主要铜矿企业、中国有色企业、铜库存、注销仓单和能源成本变量，用于刻画铜供给约束和成本变化。

在每个预测时点 \(t\)，本文分别基于各图内节点过去 60 个交易日的收益率序列计算 Spearman 相关系数，并保留每个节点相关性绝对值最高的 5 条边构造邻接矩阵。同时，对金融资产类节点，以收益率、累计收益率、波动率、成交量变化、日内振幅和均线偏离等指标构造节点属性；对库存类节点，以库存变化率、库存波动率、库存分位数、库存均线偏离和累库/去库信号构造节点属性。三个图分别输入 GAE，提取市场共振、需求预期和供给扰动特征。最终，将 GAE 图特征与 CNN 铜自身特征拼接，并输入 MLP 预测下一交易日铜价最高价和最低价。

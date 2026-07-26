"""Static configuration for the CNN + three-GAE copper model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NodeType = Literal["financial", "inventory"]

CNN_WINDOW = 20
CNN_FEATURE_DIM = 10
NODE_FEATURE_DIM = 8
CORR_WINDOW = 60
TOP_K_EDGES = 5
COPPER_NODE_NAME = "LME Copper"


@dataclass(frozen=True)
class NodeSpec:
    """A graph node and the feature recipe it should use."""

    name: str
    node_type: NodeType
    symbol: str | None = None
    description: str = ""


@dataclass(frozen=True)
class GraphSpec:
    """Fixed node set for one GAE branch."""

    name: Literal["market", "demand", "supply"]
    nodes: tuple[NodeSpec, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def copper_index(self) -> int:
        for idx, node in enumerate(self.nodes):
            if node.name == COPPER_NODE_NAME:
                return idx
        raise ValueError(f"{self.name} graph does not contain {COPPER_NODE_NAME}")


MARKET_NODES: tuple[NodeSpec, ...] = (
    NodeSpec("LME Copper", "financial", description="Target copper futures"),
    NodeSpec("COMEX Copper", "financial", description="International copper futures"),
    NodeSpec("SHFE Copper", "financial", description="China copper futures"),
    NodeSpec("Gold", "financial", description="Precious metal and dollar-priced commodity"),
    NodeSpec("Silver", "financial", description="Precious metal with industrial use"),
    NodeSpec("Crude Oil", "financial", description="Energy cost and inflation"),
    NodeSpec("Natural Gas", "financial", description="Energy price"),
    NodeSpec("Aluminum", "financial", description="Industrial metal"),
    NodeSpec("Zinc", "financial", description="Industrial metal"),
    NodeSpec("Nickel", "financial", description="Industrial metal"),
    NodeSpec("Lead", "financial", description="Industrial metal"),
    NodeSpec("Iron Ore", "financial", description="Ferrous complex and China demand"),
    NodeSpec("DXY", "financial", description="US dollar index"),
    NodeSpec("US 10Y Yield", "financial", description="US rate environment"),
    NodeSpec("VIX", "financial", description="Global risk appetite"),
    NodeSpec("S&P 500", "financial", description="Global equity risk asset"),
    NodeSpec("Nasdaq 100", "financial", description="Technology risk appetite"),
    NodeSpec("MSCI Emerging Markets", "financial", description="Emerging market demand"),
    NodeSpec("CSI 300", "financial", description="China macro demand"),
    NodeSpec("CNY/USD", "financial", description="Renminbi exchange rate"),
)

DEMAND_NODES: tuple[NodeSpec, ...] = (
    NodeSpec("LME Copper", "financial", description="Target copper futures"),
    NodeSpec("NVIDIA", "financial", description="AI hardware and data centers"),
    NodeSpec("TSMC", "financial", description="Semiconductor manufacturing"),
    NodeSpec("AMD", "financial", description="AI chips"),
    NodeSpec("Broadcom", "financial", description="AI and communications chips"),
    NodeSpec("Eaton", "financial", description="Power equipment"),
    NodeSpec("Schneider Electric", "financial", description="Electrical equipment"),
    NodeSpec("ABB", "financial", description="Electrical automation"),
    NodeSpec("Tesla", "financial", description="Electric vehicles"),
    NodeSpec("BYD", "financial", description="Electric vehicles"),
    NodeSpec("CATL / Ningde Times", "financial", description="Power batteries"),
    NodeSpec("NARI Technology", "financial", description="Grid automation"),
    NodeSpec("China XD Electric", "financial", description="Power transmission equipment"),
    NodeSpec("Shanghai Electric", "financial", description="Power equipment"),
    NodeSpec("CSI New Energy Index", "financial", description="China new energy demand"),
    NodeSpec("CSI Infrastructure Index", "financial", description="China infrastructure demand"),
    NodeSpec("CSI Real Estate Index", "financial", description="Property-chain demand"),
    NodeSpec("CSI 300", "financial", description="China equity demand expectation"),
)

SUPPLY_NODES: tuple[NodeSpec, ...] = (
    NodeSpec("LME Copper", "financial", description="Target copper futures"),
    NodeSpec("Freeport-McMoRan", "financial", description="Global copper miner"),
    NodeSpec("Southern Copper", "financial", description="Global copper miner"),
    NodeSpec("Antofagasta", "financial", description="Copper miner"),
    NodeSpec("First Quantum Minerals", "financial", description="Copper miner"),
    NodeSpec("Glencore", "financial", description="Commodities and copper mining"),
    NodeSpec("BHP", "financial", description="Diversified miner"),
    NodeSpec("Rio Tinto", "financial", description="Diversified miner"),
    NodeSpec("Anglo American", "financial", description="Diversified miner"),
    NodeSpec("Zijin Mining", "financial", description="China miner"),
    NodeSpec("Jiangxi Copper", "financial", description="Copper smelting and mining"),
    NodeSpec("Tongling Nonferrous", "financial", description="Copper smelting"),
    NodeSpec("Yunnan Copper", "financial", description="Copper smelting"),
    NodeSpec("China Molybdenum", "financial", description="Nonferrous mining"),
    NodeSpec("Crude Oil", "financial", description="Energy cost"),
    NodeSpec("Baltic Dry Index", "financial", description="Freight and global trade"),
    NodeSpec("LME Copper Inventory", "inventory", description="LME copper stock"),
    NodeSpec("SHFE Copper Inventory", "inventory", description="SHFE copper stock"),
    NodeSpec("COMEX Copper Inventory", "inventory", description="COMEX copper stock"),
    NodeSpec("LME Cancelled Warrants", "inventory", description="LME cancelled warrants"),
)


def default_graph_specs() -> dict[str, GraphSpec]:
    """Return the fixed graph specifications used by the first model version."""

    return {
        "market": GraphSpec("market", MARKET_NODES),
        "demand": GraphSpec("demand", DEMAND_NODES),
        "supply": GraphSpec("supply", SUPPLY_NODES),
    }

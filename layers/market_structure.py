"""
Layer 3: Market Structure Detection
Identifies order blocks, structure breaks, VWAP relationship, and volume profile nodes.
Analogous to geographical features that channel and concentrate weather systems.
"""

import numpy as np
import pandas as pd


class MarketStructure:
    def __init__(self, order_block_lookback: int = 20, volume_threshold: float = 1.5):
        self.order_block_lookback = order_block_lookback
        self.volume_threshold = volume_threshold

    def find_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify bullish and bearish order blocks."""
        blocks = []
        for i in range(self.order_block_lookback, len(df)):
            window = df.iloc[i - self.order_block_lookback:i]
            avg_vol = window["volume"].mean()

            candle = df.iloc[i]
            if candle["volume"] > avg_vol * self.volume_threshold:
                direction = "bullish" if candle["close"] > candle["open"] else "bearish"
                blocks.append({
                    "date": df.index[i],
                    "direction": direction,
                    "high": candle["high"],
                    "low": candle["low"],
                    "mid": (candle["high"] + candle["low"]) / 2,
                })
        return pd.DataFrame(blocks)

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling VWAP."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    def find_volume_nodes(self, df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
        """Identify high-volume price nodes (volume profile)."""
        hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
        node_prices = (edges[:-1] + edges[1:]) / 2
        return pd.DataFrame({"price": node_prices, "volume": hist}).sort_values(
            "volume", ascending=False
        )

    def score(self, df: pd.DataFrame, current_price: float) -> dict:
        """Return structural score for current price context."""
        vwap = self.calculate_vwap(df).iloc[-1]
        above_vwap = current_price > vwap

        nodes = self.find_volume_nodes(df)
        nearest_node = nodes["price"].sub(current_price).abs().idxmin()
        distance_to_node = abs(nodes.loc[nearest_node, "price"] - current_price)

        return {
            "vwap": vwap,
            "above_vwap": above_vwap,
            "nearest_volume_node": nodes.loc[nearest_node, "price"],
            "distance_to_node": distance_to_node,
        }

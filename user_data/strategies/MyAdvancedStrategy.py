import logging
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)
class MyAdvancedStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    stoploss = -1.0
    risk_per_trade = 0.01
    minimal_roi = {
        "0": 0.005,
        "5": 0.005,
        "15": 0.01,
        "120": 0.05,
    }
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True
    min_quote_volume = 100000
    min_volume = 1000
    max_open_trades = 6
    startup_candle_count = 50
    buy_rsi_threshold = 30
    sell_rsi_threshold = 65
    bb_period = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    confluence_threshold = 75
    market_volatility_threshold = 0.02
    min_profit_ratio = 2.5
    volume_spike_threshold = 2
    trend_strength_threshold = 1

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["rsi_smooth"] = df["rsi"].rolling(window=3).mean()
        df["rsi_fast"] = ta.RSI(df, timeperiod=7)
        def generate_signal(row):
            if pd.isna(row["rsi_smooth"]):
                return None
            if row["rsi_smooth"] < 30:
                return "BUY"
            elif row["rsi_smooth"] > 70:
                return "SELL"
            else:
                return None
        df["rsi_signal"] = df.apply(generate_signal, axis=1)
        macd = ta.MACD(
            df, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )
        df["macd"] = macd["macd"]
        df["macdsignal"] = macd["macdsignal"]
        df["macd_histogram"] = macd["macdhist"]
        df["macd_momentum"] = df["macd_histogram"] - df["macd_histogram"].shift(1)
        df["macd_crossover"] = (df["macd"] > df["macdsignal"]) & (
            df["macd"].shift(1) <= df["macdsignal"].shift(1)
        )
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df["close"], timeperiod=self.bb_period)
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_squeeze"] = (
            df["bb_width"] < df["bb_width"].rolling(20).mean() * 0.8
        )
        df["quote_volume"] = df["volume"] * df["close"]
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["volume_spike"] = df["volume_ratio"] > self.volume_spike_threshold
        df["ema_fast"] = ta.EMA(df["close"], timeperiod=8)
        df["ema_slow"] = ta.EMA(df["close"], timeperiod=21)
        df["ema_very_slow"] = ta.EMA(df["close"], timeperiod=50)
        df["trend"] = np.where(
            df["ema_fast"] > df["ema_slow"], 1, np.where(df["ema_fast"] < df["ema_slow"], -1, 0)
        )
        df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
        df["long_term_trend"] = np.where(df["ema_slow"] > df["ema_very_slow"], 1, -1)


        df["atr"] = ta.ATR(df, timeperiod=14)
        df["atr_normalized"] = df["atr"] / df["close"]
        df["volatility_filter"] = df["atr_normalized"] > self.market_volatility_threshold
        stoch = ta.STOCH(df)
        df["stoch_k"] = stoch["slowk"]
        df["stoch_d"] = stoch["slowd"]
        df["stoch_oversold"] = df["stoch_k"] < 20
        df["higher_low"] = (df["low"] > df["low"].shift(1)) & (
            df["low"].shift(1) < df["low"].shift(2)
        )
        df["bullish_engulfing"] = (
            (df["open"] < df["close"])
            & (df["open"].shift(1) > df["close"].shift(1))
            & (df["open"] <= df["close"].shift(1))
            & (df["close"] >= df["open"].shift(1))
        )
        df["confluence_score"] = self.calculate_enhanced_confluence_score(df)
        df["risk_reward_ratio"] = self.calculate_risk_reward(df)
        df["market_regime"] = self.detect_market_regime(df)
        return df

    def calculate_enhanced_confluence_score(self, df: DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        score += np.where(df["rsi"] < 20, 30, 0)
        score += np.where((df["rsi"] >= 20) & (df["rsi"] < 30), 20, 0)
        score += np.where((df["rsi"] >= 30) & (df["rsi"] < 40), 12, 0)
        score += np.where((df["rsi"] >= 40) & (df["rsi"] < 50), 5, 0)
        score += np.where(df["rsi_fast"] < 30, 8, 0)
        score += np.where(df["macd_crossover"], 15, 0)
        score += np.where(df["macd"] > df["macdsignal"], 10, 0)
        score += np.where(df["macd_momentum"] > 0, 8, 0)
        score += np.where(df["bb_position"] < 0.1, 20, 0)
        score += np.where((df["bb_position"] >= 0.1) & (df["bb_position"] < 0.25), 12, 0)
        score += np.where(df["bb_squeeze"], 8, 0)
        score += np.where(df["volume_spike"], 15, 0)
        score += np.where(df["volume_ratio"] > 1.5, 10, 0)
        score += np.where(df["volume_ratio"] > 1.2, 5, 0)
        score += np.where(df["trend"] == 1, 12, 0)
        score += np.where(df["long_term_trend"] == 1, 8, 0)
        score += np.where(df["trend_strength"] > self.trend_strength_threshold, 5, 0)
        score += np.where(df["stoch_oversold"], 10, 0)
        score += np.where(df["higher_low"], 8, 0)
        score += np.where(df["bullish_engulfing"], 12, 0)
        return score

    def calculate_risk_reward(self, df: DataFrame) -> pd.Series:
        target_distance = np.maximum(
            df["atr"] * 3,
            (df["bb_upper"] - df["close"]) * 0.8,
        )
        reward = target_distance / df["close"]
        risk_distance = np.maximum(
            df["atr"] * 2,
            (df["close"] - df["bb_lower"]) * 0.5,
        )
        risk = risk_distance / df["close"]
        return np.where(risk > 0, reward / risk, 0)

    def detect_market_regime(self, df: DataFrame) -> pd.Series:
        """Detect market regime: 1=trending, 0=ranging, -1=volatile"""
        adx = ta.ADX(df, timeperiod=14)
        regime = pd.Series(0, index=df.index)
        regime = np.where(adx > 25, 1, regime)
        regime = np.where(df["volatility_filter"] & (adx < 20), -1, regime)
        return regime

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata["pair"].replace("/", "")
        macd_strict_condition = (
            (df["macd"] > df["macdsignal"])
            & (df["macd"].shift(1) <= df["macdsignal"].shift(1))
            & (df["macd_histogram"] > 0.03)
            & (df["macd_momentum"] > 0)
            & (df["macd"] > df["macd"].shift(1))
            & (df["macdsignal"] > df["macdsignal"].shift(1))
            & (df["macd"] > 0)
            & (df["macdsignal"] > 0)
        )
        df["condition_score"] = (
            (df["rsi_smooth"] < self.buy_rsi_threshold).astype(int)
            + macd_strict_condition.astype(int)
            + (df["quote_volume"] > self.min_quote_volume).astype(int)
            + (df["volume_ratio"] > self.volume_spike_threshold).astype(int)
            + (df["confluence_score"] > self.confluence_threshold).astype(int)
            + (df["risk_reward_ratio"] > self.min_profit_ratio).astype(int)
            + (df["market_regime"] > 0).astype(int)
            + (df["trend_strength"] > self.trend_strength_threshold).astype(int)
            + (df["bb_position"] < 0.95).astype(int)
        )
        df["buy"] = 0
        condition = (df["condition_score"] >= 6) & (df["rsi_smooth"] <= self.buy_rsi_threshold)
        df.loc[condition, "buy"] = 1


        overlap = df[(df["condition_score"] >= 6) & (df["rsi_smooth"] <= self.buy_rsi_threshold)]
        logger.info(f"Overlapping candles passing buy criteria {pair} : {len(overlap)}")

        logger.info(
            f"Buy condition breakdown: RSI < {self.buy_rsi_threshold} passes: {(df['rsi_smooth'] <= self.buy_rsi_threshold).sum()}"  # noqa: E501
        )
        logger.info(
            f"MACD condition passes: {((df['macd'] > df['macdsignal']) | (df['macd_crossover'])).sum()}"  # noqa: E501
        )
        logger.info(
            f"Volume spike passes: {(df['volume_ratio'] > self.volume_spike_threshold).sum()}"
        )
        logger.info(
            f"Confluence passes: {(df['confluence_score'] > self.confluence_threshold).sum()}"
        )
        logger.info(
            f"Risk/reward passes: {(df['risk_reward_ratio'] > self.min_profit_ratio).sum()}"
        )
        logger.info(f"Market regime passes: {(df['market_regime'] > 0).sum()}")
        logger.info(f"Total condition_score passes >=6: {(df['condition_score'] >= 6).sum()}")

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (df["rsi"] > self.sell_rsi_threshold) | (df["rsi_fast"] > 75),

            (df["macd"] < df["macdsignal"]),

            (df["confluence_score"] < self.confluence_threshold * 0.8),
            (
                (df["bb_position"] > 0.9)
                | (df["trend"] == -1)
                | (df["stoch_k"] > 80)
            ),
        ]
        df.loc[reduce(lambda x, y: x & y, conditions), "sell"] = 1
        return df

        #   def custom_stoploss(
        #       self,
        #      pair: str,
        #     trade: Trade,
        #    current_time: datetime,
        #   current_rate: float,
        #     current_profit: float,
        #    **kwargs,
        # ) -> float:
        """
        Dynamic stop loss based on ATR and trade duration
        """

    # if current_profit < -0.02:  # Hard stop at -2%
    #    return -0.02

    # Get the latest dataframe
    # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    # if dataframe.empty:
    #    return self.stoploss

    # Get current ATR
    # current_atr = dataframe["atr_normalized"].iloc[-1]

    # Calculate dynamic stop based on ATR
    # if current_profit > 0.01:  # If in profit > 1%
    # Tighter stop using ATR
    #   dynamic_stop = -(current_atr * 1.5)
    # else:
    #     # Wider stop for early stages
    ##       dynamic_stop = -(current_atr * 2.5)

    ##  return max(dynamic_stop, self.stoploss)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Dynamic stoploss:
        - Hard stop at -5%
        - ATR-based dynamic stoploss for adaptive risk management
        """
        if current_profit < -0.05:
            return -0.05

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
        current_atr = dataframe["atr_normalized"].iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            return self.stoploss
        if current_profit > 0.02:

            dynamic_stop = -(current_atr * 1.5)
        else:

            dynamic_stop = -(current_atr * 2.5)
        return max(dynamic_stop, self.stoploss)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Additional trade entry confirmation
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return True
        if dataframe["volatility_filter"].iloc[-1]:
            logger.info(f"Skipping {pair} due to high volatility")
            return False
        current_confluence = dataframe["confluence_score"].iloc[-1]
        if current_confluence < self.confluence_threshold * 0.9:
            logger.info(f"Skipping {pair} due to low confluence score: {current_confluence}")
            return False
        return True

plot_config = {
    "main_plot": {
        "close": {"color": "blue"},
        "ema_fast": {"color": "orange"},
        "ema_slow": {"color": "red"},
        "ema_very_slow": {"color": "purple"},
        "bb_upper": {"color": "grey"},
        "bb_middle": {"color": "lightgrey"},
        "bb_lower": {"color": "grey"},
    },
    "subplots": {
        "RSI": {
            "rsi": {"color": "blue"},
            "rsi_smooth": {"color": "purple"},
            "rsi_fast": {"color": "orange"},
        },
        "MACD": {
            "macd": {"color": "blue"},
            "macdsignal": {"color": "red"},
            "macd_histogram": {"color": "grey"},
        },
        "Bollinger": {
            "bb_width": {"color": "green"},
            "bb_position": {"color": "brown"},
        },
        "Volume": {
            "volume": {"color": "grey"},
            "volume_sma": {"color": "orange"},
            "volume_ratio": {"color": "blue"},
        },
        "Stochastic": {
            "stoch_k": {"color": "blue"},
            "stoch_d": {"color": "red"},
        },
        "Confluence": {
            "confluence_score": {"color": "black"},
        },
        "Risk/Reward": {
            "risk_reward_ratio": {"color": "gold"},
        },
        "Market Regime": {
            "market_regime": {"color": "cyan"},
        },
    },
}

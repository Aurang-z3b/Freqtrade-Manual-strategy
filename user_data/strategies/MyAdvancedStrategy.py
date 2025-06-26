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
    
    # Strategy parameters
    timeframe = "5m"
    minimal_roi = {
        "0": 0.05,
        "30": 0.05,
        "60": 0.05,
        "120": 0.01,
        "240": 0.01
    }
    
    # Stoploss
    stoploss = -0.03
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    
    # Risk Management
    risk_per_trade = 0.01
    max_open_trades = 2
    
    # Strategy Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Volume Analysis Parameters
    volume_trend_window = 20
    volume_profile_window = 20
    volume_spike_threshold = 2.0
    
    # Market Profile Parameters
    market_profile_window = 20
    
    # Volatility Parameters
    volatility_window = 20
    market_volatility_threshold = 0.02
    
    # Initialize indicators
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.info('Initializing MyAdvancedStrategy')
        
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for pair {metadata.get('pair', 'unknown')}")
                return df
            
            # Handle NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill()
            
            # Verify required OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. DataFrame columns: {df.columns.tolist()}")
                return df
            
            # Calculate RSI with NaN handling
            df["rsi"] = ta.RSI(df["close"], timeperiod=14)
            df["rsi_smooth"] = df["rsi"].rolling(window=3).mean()
            df["rsi_fast"] = ta.RSI(df["close"], timeperiod=7)
            
            # Generate RSI signals with NaN handling
            def generate_signal(row):
                if pd.isna(row["rsi_smooth"]):
                    return None
                if row["rsi_smooth"] < 30:
                    return "BUY"
                elif row["rsi_smooth"] > 70:
                    return "SELL"
                return None
            
            df["rsi_signal"] = df.apply(generate_signal, axis=1)
            
            # Calculate MACD with NaN handling
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
            
            # Volume Analysis
            df["volume_ma"] = df["volume"].rolling(window=self.volume_trend_window).mean()
            df["volume_spike"] = df["volume"] / df["volume_ma"]
            df["volume_trend"] = df["volume"].pct_change()
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
            df["volume_spike"] = df["volume_ratio"] > self.volume_spike_threshold
            
            # Volume Profile
            if 'volume' in df.columns:
                df["volume_profile"] = df["volume"].rolling(window=self.volume_profile_window).mean()
                df["volume_profile_std"] = df["volume"].rolling(window=self.volume_profile_window).std()
            else:
                logger.warning("Volume column not found for volume profile analysis")
            
            # Market Volatility
            df["volatility"] = df["close"].rolling(window=self.volatility_window).std() / df["close"].rolling(window=self.volatility_window).mean()
            df["volatility_filter"] = df["volatility"] > self.market_volatility_threshold
            
            # Multiple Timeframe Analysis
            # Add 15m timeframe analysis
            # Ensure we have a proper datetime index
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
            
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate 15m indicators
            df_15m["rsi_15m"] = ta.RSI(df_15m["close"], timeperiod=14)
            df_15m["macd_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macd"]
            df_15m["macdsignal_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macdsignal"]
            
            # Merge 15m indicators back to 5m timeframe
            df = df.merge(df_15m, how='left', left_index=True, right_index=True)
            
            # Market Profile Analysis
            if 'high' in df.columns and 'low' in df.columns:
                df["market_profile_high"] = df["high"].rolling(window=self.market_profile_window).max()
                df["market_profile_low"] = df["low"].rolling(window=self.market_profile_window).min()
                df["market_profile_range"] = df["market_profile_high"] - df["market_profile_low"]
            else:
                logger.warning("Required OHLCV columns not found for market profile analysis")
            
            # Orderbook Analysis (if available)
            if 'volume' in df.columns and 'volume_ma' in df.columns:
                df["orderbook_imbalance"] = np.where(
                    df["volume"] > df["volume_ma"],
                    df["volume"] / df["volume_ma"],
                    0
                )
            else:
                logger.warning("Volume columns not found for orderbook analysis")
            
            # Trend Analysis
            df["ema_fast"] = ta.EMA(df["close"], timeperiod=8)
            df["ema_slow"] = ta.EMA(df["close"], timeperiod=21)
            df["ema_very_slow"] = ta.EMA(df["close"], timeperiod=50)
            
            # Market Regime
            df["market_regime"] = np.where(
                df["ema_fast"] > df["ema_slow"],
                "Trending Up",
                np.where(
                    df["ema_fast"] < df["ema_slow"],
                    "Trending Down",
                    "Sideways"
                )
            )
            
            # Risk/Reward Analysis
            df["risk_reward_ratio"] = df["ema_fast"] / df["ema_slow"]
            
            return df
            
        except KeyError as e:
            logger.error(f"KeyError: Column '{str(e)}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Unexpected error in populate_indicators: {str(e)}")
            return df
    
    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Based on TA indicators, populates the buy signal for the given dataframe
        :param df: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column"""
        df['buy'] = (
            # RSI conditions
            (df['rsi'] < 30) &
            (df['rsi'].shift(1) > 30) &  # RSI crossed below 30
            
            # MACD conditions
            (df['macd'] > df['macdsignal']) &
            (df['macd'].shift(1) <= df['macdsignal'].shift(1)) &  # MACD crossed above signal line
            
            # Volume conditions
            (df['volume'] > df['volume_ma']) &  # Volume above its MA
            
            # Trend conditions
            (df['ema_fast'] > df['ema_slow']) &  # Fast EMA above slow EMA
            
            # Risk/Reward conditions
            (df['risk_reward_ratio'] > 1.5)  # Good risk/reward ratio
        ).astype('int')
        
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Based on TA indicators, populates the sell signal for the given dataframe
        :param df: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column"""
        df['sell'] = (
            # RSI conditions
            (df['rsi'] > 70) &
            (df['rsi'].shift(1) < 70) &  # RSI crossed above 70
            
            # MACD conditions
            (df['macd'] < df['macdsignal']) &
            (df['macd'].shift(1) >= df['macdsignal'].shift(1)) &  # MACD crossed below signal line
            
            # Volume conditions
            (df['volume'] < df['volume_ma']) &  # Volume below its MA
            
            # Trend conditions
            (df['ema_fast'] < df['ema_slow']) &  # Fast EMA below slow EMA
            
            # Risk/Reward conditions
            (df['risk_reward_ratio'] < 0.5)  # Poor risk/reward ratio
        ).astype('int')
        
        return df


logger = logging.getLogger(__name__)
class MyAdvancedStrategy(IStrategy):  # noqa: F811
    INTERFACE_VERSION = 3
    timeframe = "5m"
    
    # Risk Management Parameters
    stoploss = -0.03  # 2% stoploss
    risk_per_trade = 0.01  # 1% risk per trade
    minimal_roi = {
        "0": 0.05,       # 1% for first 30 minutes
        "30": 0.05,    # 0.75% after 30 minutes
        "60": 0.05,     # 0.5% after 1 hour
        "120": 0.01,    # 0.3% after 2 hours
        "240": 0.01     # 0.1% after 4 hours
    }
    
    # Position Sizing Parameters
    max_open_trades = 2  # Further reduced to 2
    max_drawdown = 0.05  # Reduced to 5% drawdown
    min_position_size = 0.02  # Increased to 2%
    max_position_size = 0.03  # Reduced to 3%
    
    # Strategy Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Volume Analysis Parameters
    volume_trend_window = 20
    volume_profile_window = 20
    volume_spike_threshold = 2.0
    
    # Market Profile Parameters
    market_profile_window = 20
    
    # Volatility Parameters
    volatility_window = 20
    market_volatility_threshold = 0.02
    
    # Initialize indicators
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.info('Initializing MyAdvancedStrategy')
        
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for pair {metadata.get('pair', 'unknown')}")
                return df
            
            # Handle NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill()
            
            # Verify required OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. DataFrame columns: {df.columns.tolist()}")
                return df
            
            # Calculate RSI with NaN handling
            df["rsi"] = ta.RSI(df["close"], timeperiod=14)
            df["rsi_smooth"] = df["rsi"].rolling(window=3).mean()
            df["rsi_fast"] = ta.RSI(df["close"], timeperiod=7)
            
            # Generate RSI signals with NaN handling
            def generate_signal(row):
                if pd.isna(row["rsi_smooth"]):
                    return None
                if row["rsi_smooth"] < 30:
                    return "BUY"
                elif row["rsi_smooth"] > 70:
                    return "SELL"
                return None
            
            df["rsi_signal"] = df.apply(generate_signal, axis=1)
            
            # Calculate MACD with NaN handling
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
            
            # Volume Analysis
            df["volume_ma"] = df["volume"].rolling(window=self.volume_trend_window).mean()
            df["volume_spike"] = df["volume"] / df["volume_ma"]
            df["volume_trend"] = df["volume"].pct_change()
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
            df["volume_spike"] = df["volume_ratio"] > self.volume_spike_threshold
            
            # Volume Profile
            if 'volume' in df.columns:
                df["volume_profile"] = df["volume"].rolling(window=self.volume_profile_window).mean()
                df["volume_profile_std"] = df["volume"].rolling(window=self.volume_profile_window).std()
            else:
                logger.warning("Volume column not found for volume profile analysis")
            
            # Market Volatility
            df["volatility"] = df["close"].rolling(window=self.volatility_window).std() / df["close"].rolling(window=self.volatility_window).mean()
            df["volatility_filter"] = df["volatility"] > self.market_volatility_threshold
            
            # Multiple Timeframe Analysis
            # Add 15m timeframe analysis
            # Ensure we have a proper datetime index
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
            
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate 15m indicators
            df_15m["rsi_15m"] = ta.RSI(df_15m["close"], timeperiod=14)
            df_15m["macd_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macd"]
            df_15m["macdsignal_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macdsignal"]
            
            # Merge 15m indicators back to 5m timeframe
            df = df.merge(df_15m, how='left', left_index=True, right_index=True)
            
            # Market Profile Analysis
            if 'high' in df.columns and 'low' in df.columns:
                df["market_profile_high"] = df["high"].rolling(window=self.market_profile_window).max()
                df["market_profile_low"] = df["low"].rolling(window=self.market_profile_window).min()
                df["market_profile_range"] = df["market_profile_high"] - df["market_profile_low"]
            else:
                logger.warning("Required OHLCV columns not found for market profile analysis")
            
            # Orderbook Analysis (if available)
            if 'volume' in df.columns and 'volume_ma' in df.columns:
                df["orderbook_imbalance"] = np.where(
                    df["volume"] > df["volume_ma"],
                    df["volume"] / df["volume_ma"],
                    0
                )
            else:
                logger.warning("Volume columns not found for orderbook analysis")
            
            # Trend Analysis
            df["ema_fast"] = ta.EMA(df["close"], timeperiod=8)
            df["ema_slow"] = ta.EMA(df["close"], timeperiod=21)
            df["ema_very_slow"] = ta.EMA(df["close"], timeperiod=50)
            
            # Market Regime
            df["market_regime"] = np.where(
                df["ema_fast"] > df["ema_slow"],
                "Trending Up",
                np.where(
                    df["ema_fast"] < df["ema_slow"],
                    "Trending Down",
                    "Sideways"
                )
            )
            
            # Risk/Reward Analysis
            df["risk_reward_ratio"] = df["ema_fast"] / df["ema_slow"]
            
            return df
            
        except KeyError as e:
            logger.error(f"KeyError: Column '{str(e)}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Unexpected error in populate_indicators: {str(e)}")
            return df
    
    # Trailing Stop Parameters
    trailing_stop = True
    trailing_stop_positive = 0.025  # 2.5% trailing stop once profit is reached
    trailing_stop_positive_offset = 0.05  # 5% trailing stop offset
    trailing_only_offset_is_reached = True
    
    # Trailing Stop Adjustments
    trailing_stop_adjustment = 0.005  # 0.5% adjustment factor for market conditions
    trailing_stop_max_offset = 0.10  # Maximum 10% trailing stop offset
    trailing_stop_min_offset = 0.02  # Minimum 2% trailing stop offset
    
    # Market Condition Multipliers for Trailing Stop
    trending_market_multiplier = 1.2  # 20% increase in trailing stop offset in trending markets
    volatile_market_multiplier = 0.8  # 20% decrease in trailing stop offset in volatile markets
    
    # Trailing Stop Timing Parameters
    min_trailing_duration = 30  # Minimum 30 minutes before trailing stop activates
    max_trailing_duration = 120  # Maximum 120 minutes trailing duration
    
    # Additional Risk Parameters
    max_position_risk = 0.02  # 2% maximum risk per position
    max_total_risk = 0.03    # 3% maximum total risk
    risk_reward_ratio = 2.0   # Minimum 2:1 risk/reward ratio
    
    # Market Condition Parameters
    trending_market_multiplier = 1.2  # 20% increase in position size in trending markets
    volatile_market_multiplier = 0.8  # 20% decrease in position size in volatile markets
    
    # Position Sizing Parameters
    initial_position_size = 0.02  # 2% initial position size
    position_size_step = 0.01     # 1% step size for position scaling
    max_position_scaling = 3      # Maximum 3x scaling of initial position
    
    # Time-based Parameters
    cooldown_period = 60  # 60 minutes cooldown after loss
    max_consecutive_losses = 3    # Maximum 3 consecutive losses before cooldown
    recovery_period = 120  # 120 minutes recovery period after cooldown
    
    # Indicator Parameters
    min_quote_volume = 100000  # Increased to filter out low volume trades
    min_volume = 1000  # Increased for better volume confirmation
    startup_candle_count = 150  # Increased for better indicator initialization
    
    # Volume Analysis Parameters
    volume_spike_threshold = 2.0  # Increased to 200% for more significant volume spikes
    volume_trend_window = 12  # Window for volume trend analysis
    volume_profile_window = 24  # Window for volume profile analysis
    
    # Market Volatility Parameters
    volatility_window = 24  # Window for volatility calculation
    max_volatility_threshold = 0.02  # Maximum acceptable volatility
    
    # Confirmation Signal Parameters
    confirmation_window = 24  # Window for multiple timeframe analysis
    orderbook_depth = 20  # Depth for orderbook analysis
    market_profile_window = 24  # Window for market profile analysis
    
    # RSI Parameters
    buy_rsi_threshold = 35  # Slightly higher than default
    sell_rsi_threshold = 70  # Slightly lower than default
    
    # BB Parameters
    bb_period = 25  # Increased from 20 for better smoothing
    
    # MACD Parameters
    macd_fast = 14  # Slightly slower than default
    macd_slow = 28  # Slightly slower than default
    macd_signal = 10  # Slightly slower than default
    
    # Other Parameters
    confluence_threshold = 80  # Increased for stronger signals
    market_volatility_threshold = 0.03  # Increased for more aggressive trading
    min_profit_ratio = 2.0  # Reduced from 2.5 for more trades
    volume_spike_threshold = 1.5  # Reduced for more sensitive volume detection
    trend_strength_threshold = 2  # Increased for stronger trend confirmation
    
    # Plot Configuration
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

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
        try:
            # Handle NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill()
            
            # Verify required OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. DataFrame columns: {df.columns.tolist()}")
                return df

        except Exception as e:
            logger.error(f"Unexpected error in populate_indicators: {str(e)}")
            return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Based on TA indicators, populates the buy signal for the given dataframe
        :param df: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column"""
        df['buy'] = (
            # RSI conditions
            (df['rsi'] < 30) &
            (df['rsi'].shift(1) > 30) &  # RSI crossed below 30
            
            # MACD conditions
            (df['macd'] > df['macdsignal']) &
            (df['macd'].shift(1) <= df['macdsignal'].shift(1)) &  # MACD crossed above signal line
            
            # Volume conditions
            (df['volume'] > df['volume_ma']) &  # Volume above its MA
            
            # Trend conditions
            (df['ema_fast'] > df['ema_slow']) &  # Fast EMA above slow EMA
            
            # Risk/Reward conditions
            (df['risk_reward_ratio'] > 1.5)  # Good risk/reward ratio
        ).astype('int')
        
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Based on TA indicators, populates the sell signal for the given dataframe
        :param df: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column"""
        df['sell'] = (
            # RSI conditions
            (df['rsi'] > 70) &
            (df['rsi'].shift(1) < 70) &  # RSI crossed above 70
            
            # MACD conditions
            (df['macd'] < df['macdsignal']) &
            (df['macd'].shift(1) >= df['macdsignal'].shift(1)) &  # MACD crossed below signal line
            
            # Volume conditions
            (df['volume'] < df['volume_ma']) &  # Volume below its MA
            
            # Trend conditions
            (df['ema_fast'] < df['ema_slow']) &  # Fast EMA below slow EMA
            
            # Risk/Reward conditions
            (df['risk_reward_ratio'] < 0.5)  # Poor risk/reward ratio
        ).astype('int')
        
        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
        try:
            # Handle NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill()
            
            # Verify required OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. DataFrame columns: {df.columns.tolist()}")
                return df

            # Calculate RSI with NaN handling
            df["rsi"] = ta.RSI(df["close"], timeperiod=14)
            df["rsi_smooth"] = df["rsi"].rolling(window=3).mean()
            df["rsi_fast"] = ta.RSI(df["close"], timeperiod=7)
            
            # Generate RSI signals with NaN handling
            def generate_signal(row):
                if pd.isna(row["rsi_smooth"]):
                    return None
                if row["rsi_smooth"] < 30:
                    return "BUY"
                elif row["rsi_smooth"] > 70:
                    return "SELL"
                return None
            
            df["rsi_signal"] = df.apply(generate_signal, axis=1)
            
            # Calculate MACD with NaN handling
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
            
            # Volume Analysis
            df["volume_ma"] = df["volume"].rolling(window=self.volume_trend_window).mean()
            df["volume_spike"] = df["volume"] / df["volume_ma"]
            df["volume_trend"] = df["volume"].pct_change()
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
            df["volume_spike"] = df["volume_ratio"] > self.volume_spike_threshold
            
            # Volume Profile
            if 'volume' in df.columns:
                df["volume_profile"] = df["volume"].rolling(window=self.volume_profile_window).mean()
                df["volume_profile_std"] = df["volume"].rolling(window=self.volume_profile_window).std()
            else:
                logger.warning("Volume column not found for volume profile analysis")
            
            # Market Volatility
            df["volatility"] = df["close"].rolling(window=self.volatility_window).std() / df["close"].rolling(window=self.volatility_window).mean()
            df["volatility_filter"] = df["volatility"] > self.market_volatility_threshold
            
            # Multiple Timeframe Analysis
            # Add 15m timeframe analysis
            # Ensure we have a proper datetime index
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
            
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate 15m indicators
            df_15m["rsi_15m"] = ta.RSI(df_15m["close"], timeperiod=14)
            df_15m["macd_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macd"]
            df_15m["macdsignal_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macdsignal"]
            
            # Merge 15m indicators back to 5m timeframe
            df = df.merge(df_15m, how='left', left_index=True, right_index=True)
            
            # Market Profile Analysis
            if 'high' in df.columns and 'low' in df.columns:
                df["market_profile_high"] = df["high"].rolling(window=self.market_profile_window).max()
                df["market_profile_low"] = df["low"].rolling(window=self.market_profile_window).min()
                df["market_profile_range"] = df["market_profile_high"] - df["market_profile_low"]
            else:
                logger.warning("Required OHLCV columns not found for market profile analysis")
            
            # Orderbook Analysis (if available)
            if 'volume' in df.columns and 'volume_ma' in df.columns:
                df["orderbook_imbalance"] = np.where(
                    df["volume"] > df["volume_ma"],
                    df["volume"] / df["volume_ma"],
                    0
                )
            else:
                logger.warning("Volume columns not found for orderbook analysis")
            
            # Trend Analysis
            df["ema_fast"] = ta.EMA(df["close"], timeperiod=8)
            df["ema_slow"] = ta.EMA(df["close"], timeperiod=21)
            df["ema_very_slow"] = ta.EMA(df["close"], timeperiod=50)
            
            # Market Regime
            df["market_regime"] = np.where(
                df["ema_fast"] > df["ema_slow"],
                "Trending Up",
                np.where(
                    df["ema_fast"] < df["ema_slow"],
                    "Trending Down",
                    "Sideways"
                )
            )
            
            # Risk/Reward Analysis
            df["risk_reward_ratio"] = df["ema_fast"] / df["ema_slow"]
            
            return df
            
        except KeyError as e:
            logger.error(f"KeyError: Column '{str(e)}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Unexpected error in populate_indicators: {str(e)}")
            return df
        df["rsi_smooth"] = df["rsi"].rolling(window=3).mean()
        df["rsi_fast"] = ta.RSI(df, timeperiod=7)
        
        # Generate RSI signals with NaN handling
        def generate_signal(row):
            if pd.isna(row["rsi_smooth"]):
                return None
            if row["rsi_smooth"] < 30:
                return "BUY"
            elif row["rsi_smooth"] > 70:
                return "SELL"
            return None
        
        df["rsi_signal"] = df.apply(generate_signal, axis=1)
        
        # Calculate MACD with NaN handling
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
        
        # Volume Analysis
        df["volume_ma"] = df["volume"].rolling(window=self.volume_trend_window).mean()
        df["volume_spike"] = df["volume"] / df["volume_ma"]
        df["volume_trend"] = df["volume"].pct_change()
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
        df["volume_spike"] = df["volume_ratio"] > self.volume_spike_threshold
        
        # Volume Profile
        if 'volume' in df.columns:
            df["volume_profile"] = df["volume"].rolling(window=self.volume_profile_window).mean()
            df["volume_profile_std"] = df["volume"].rolling(window=self.volume_profile_window).std()
        else:
            logger.warning("Volume column not found for volume profile analysis")
        
        # Market Volatility
        df["volatility"] = df["close"].rolling(window=self.volatility_window).std() / df["close"].rolling(window=self.volatility_window).mean()
        df["volatility_filter"] = df["volatility"] > self.market_volatility_threshold
        
        # Multiple Timeframe Analysis
        # Add 15m timeframe analysis
        # Ensure we have a proper datetime index
        df.index = pd.to_datetime(df.index)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 15m indicators
        df_15m["rsi_15m"] = ta.RSI(df_15m, timeperiod=14)
        df_15m["macd_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macd"]
        df_15m["macdsignal_15m"] = ta.MACD(df_15m, fastperiod=12, slowperiod=26, signalperiod=9)["macdsignal"]
        
        # Merge 15m indicators back to 5m timeframe
        df = df.merge(df_15m, how='left', left_index=True, right_index=True)
        
        # Market Profile Analysis
        if 'high' in df.columns and 'low' in df.columns:
            df["market_profile_high"] = df["high"].rolling(window=self.market_profile_window).max()
            df["market_profile_low"] = df["low"].rolling(window=self.market_profile_window).min()
            df["market_profile_range"] = df["market_profile_high"] - df["market_profile_low"]
        else:
            logger.warning("Required OHLCV columns not found for market profile analysis")
        
        # Orderbook Analysis (if available)
        if 'volume' in df.columns and 'volume_ma' in df.columns:
            df["orderbook_imbalance"] = np.where(
                df["volume"] > df["volume_ma"],
                df["volume"] / df["volume_ma"],
                0
            )
        else:
            logger.warning("Volume columns not found for orderbook analysis")
        
        # Trend Analysis
        df["ema_fast"] = ta.EMA(df["close"], timeperiod=8)
        df["ema_slow"] = ta.EMA(df["close"], timeperiod=21)
        df["ema_very_slow"] = ta.EMA(df["close"], timeperiod=50)
        
        # Avoid division by zero
        df["trend_strength"] = np.where(
            df["ema_slow"] != 0,
            abs(df["ema_fast"] - df["ema_slow"]) / df["ema_slow"],
            0
        )
        
        df["trend"] = np.where(
            df["ema_fast"] > df["ema_slow"], 1, 
            np.where(df["ema_fast"] < df["ema_slow"], -1, 0)
        )
        df["long_term_trend"] = np.where(df["ema_slow"] > df["ema_very_slow"], 1, -1)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df["close"], timeperiod=self.bb_period)
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(20).mean() * 0.8
        
        # Quote Volume
        df["quote_volume"] = df["volume"] * df["close"]
        
        # ATR and Volatility
        df["atr"] = ta.ATR(df, timeperiod=14)
        df["atr_normalized"] = df["atr"] / df["close"]
        
        # Stochastic
        stoch = ta.STOCH(df)
        df["stoch_k"] = stoch["slowk"]
        df["stoch_d"] = stoch["slowd"]
        df["stoch_oversold"] = df["stoch_k"] < 20
        
        # Pattern Analysis
        df["higher_low"] = (df["low"] > df["low"].shift(1)) & (
            df["low"].shift(1) < df["low"].shift(2)
        )
        df["bullish_engulfing"] = (
            (df["open"] < df["close"])
            & (df["open"].shift(1) > df["close"].shift(1))
            & (df["open"] <= df["close"].shift(1))
            & (df["close"] >= df["open"].shift(1))
        )
        
        # Confluence Score
        df["confluence_score"] = self.calculate_enhanced_confluence_score(df)
        
        # Risk/Reward
        df["risk_reward_ratio"] = self.calculate_risk_reward(df)
        
        # Market Regime
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

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
        # Calculate bearish conditions
        strong_bearish_candle = (df['close'] < df['open']) & ((df['open'] - df['close']) / df['open'] > 0.005)
        prev_bearish = (df['close'].shift(1) < df['open'].shift(1))
        curr_not_strong_bearish = ((df['close'] >= df['open']) | ((df['open'] - df['close']) / df['open'] < 0.005))

        # Calculate MACD conditions
        macd_strict_condition = (
            (df["macd"] > df["macdsignal"])
            & (df["macd"].shift(1) <= df["macdsignal"].shift(1))
            & (df["macd_histogram"] > 0.03)
            & (df["macd_momentum"] > 0)
            & (df["macd"] > df["macd"].shift(1))
        )

        # Calculate weighted condition score
        conditions = {
            "rsi": (df["rsi_smooth"] < self.buy_rsi_threshold).astype(int) * 2,
            "macd": macd_strict_condition.astype(int) * 3,
            "volume": (df["volume"] > self.min_volume).astype(int) * 2,
            "confluence": (df["confluence_score"] > self.confluence_threshold).astype(int) * 3,
            "risk_reward": (df["risk_reward_ratio"] > self.min_profit_ratio).astype(int) * 2,
            "market_regime": (df["market_regime"] > 0).astype(int) * 1,
            "trend_strength": (df["trend_strength"] > self.trend_strength_threshold).astype(int) * 2,
            "bb_position": (df["bb_position"] < 0.95).astype(int) * 1,
            "long_term_trend": (df["long_term_trend"] == 1).astype(int) * 1,
            "volume_spike": (df["volume_spike"] > self.volume_spike_threshold).astype(int) * 2,
            "volatility": (df["volatility"] < self.max_volatility_threshold).astype(int) * 2,
            "rsi_15m": (df["rsi_15m"] < 50).astype(int) * 2,
            "macd_15m": (df["macd_15m"] > df["macdsignal_15m"]).astype(int) * 2,
            "market_profile": ((df["close"] > df["market_profile_low"]) & (df["close"] < df["market_profile_high"])).astype(int) * 2,
            "orderbook": (df["orderbook_imbalance"] > 1.5).astype(int) * 2
        }
        
        # Calculate total score with weights
        df["condition_score"] = sum(conditions.values())
        
        # Normalize score to 0-10 range
        df["condition_score"] = df["condition_score"] / 30 * 10  # Adjusted divisor for new conditions
        
        # Add position sizing based on score
        df["position_size"] = np.where(
            df["condition_score"] > 7,
            1.0,
            df["condition_score"] / 7
        )

        # Create buy condition
        condition = (
            (df["condition_score"] >= 7)  # High confidence score
            & macd_strict_condition
            & (~strong_bearish_candle)  # Don't buy on strong bearish candle
            & (curr_not_strong_bearish)  # Current candle shows strength
            & (prev_bearish)  # Previous candle was bearish (dump candle)
            & (df["volume"] > self.min_volume)  # Volume check
            & (df["volatility"] < self.max_volatility_threshold)  # Volatility check
            & (df["rsi_15m"] < 50)  # 15m RSI confirmation
            & (df["macd_15m"] > df["macdsignal_15m"])  # 15m MACD confirmation
            & (df["close"] > df["market_profile_low"])  # Above recent low
            & (df["close"] < df["market_profile_high"])  # Below recent high
            & (df["orderbook_imbalance"] > 1.5)  # Significant buying pressure
        )

        df.loc[condition, "buy"] = 1
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
        conditions = [
            # Volume Analysis
            df["volume"] > self.min_volume,
            df["volume_trend"] < 0,
            
            # Market Volatility Check
            df["volatility"] < self.max_volatility_threshold,
            
            # Multiple Timeframe Confirmation
            df["rsi_15m"] > 50,  # 15m timeframe confirmation
            df["macd_15m"] < df["macdsignal_15m"],  # 15m MACD confirmation
            
            # Market Profile Analysis
            df["close"] > df["market_profile_high"] * 0.95,  # Near recent high
            
            # Orderbook Analysis
            df["orderbook_imbalance"] < 0.5,  # Significant selling pressure
            
            # Existing Sell Conditions
            df["rsi"] > self.sell_rsi_threshold,
            df["macd_histogram"] < 0,
            df["trend"] == -1,
            df["long_term_trend"] == -1
        ]
        
        df["sell"] = reduce(lambda x, y: x & y, conditions)
        
        return df
        # Calculate bearish conditions
        strong_bearish_candle = (df['close'] < df['open']) & ((df['open'] - df['close']) / df['open'] > 0.005)
        prev_bearish = (df['close'].shift(1) < df['open'].shift(1))
        curr_not_strong_bearish = ((df['close'] >= df['open']) | ((df['open'] - df['close']) / df['open'] < 0.005))

        # Calculate MACD conditions
        macd_strict_condition = (
            (df["macd"] > df["macdsignal"])
            & (df["macd"].shift(1) <= df["macdsignal"].shift(1))
            & (df["macd_histogram"] > 0.03)
            & (df["macd_momentum"] > 0)
            & (df["macd"] > df["macd"].shift(1))
        )

        # Calculate weighted condition score
        conditions = {
            "rsi": (df["rsi_smooth"] < self.buy_rsi_threshold).astype(int) * 2,
            "macd": macd_strict_condition.astype(int) * 3,
            "volume": (df["quote_volume"] > self.min_quote_volume).astype(int) * 2,
            "confluence": (df["confluence_score"] > self.confluence_threshold).astype(int) * 3,
            "risk_reward": (df["risk_reward_ratio"] > self.min_profit_ratio).astype(int) * 2,
            "market_regime": (df["market_regime"] > 0).astype(int) * 1,
            "trend_strength": (df["trend_strength"] > self.trend_strength_threshold).astype(int) * 2,
            "bb_position": (df["bb_position"] < 0.95).astype(int) * 1,
            "long_term_trend": (df["long_term_trend"] == 1).astype(int) * 1
        }
        
        # Calculate total score with weights
        df["condition_score"] = sum(conditions.values())
        
        # Normalize score to 0-10 range
        df["condition_score"] = df["condition_score"] / 15 * 10
        
        # Add position sizing based on score
        df["position_size"] = np.where(
            df["condition_score"] > 7,
            1.0,
            df["condition_score"] / 7
        )

        # Create buy condition
        condition = (
            (df["condition_score"] >= 7)
            & (df["rsi_smooth"] <= self.buy_rsi_threshold)
            & (~strong_bearish_candle)
            & (curr_not_strong_bearish)
            & (prev_bearish)
        )

        df.loc[condition, "buy"] = 1
        return df

# Calculate weighted condition score
        conditions = {
            "rsi": (df["rsi_smooth"] < self.buy_rsi_threshold).astype(int) * 2,
            "macd": macd_strict_condition.astype(int) * 3,
            "volume": (df["quote_volume"] > self.min_quote_volume).astype(int) * 2,
            "confluence": (df["confluence_score"] > self.confluence_threshold).astype(int) * 3,
            "risk_reward": (df["risk_reward_ratio"] > self.min_profit_ratio).astype(int) * 2,
            "market_regime": (df["market_regime"] > 0).astype(int) * 1,
            "trend_strength": (df["trend_strength"] > self.trend_strength_threshold).astype(int) * 2,
            "bb_position": (df["bb_position"] < 0.95).astype(int) * 1,
            "long_term_trend": (df["long_term_trend"] == 1).astype(int) * 1
        }
        
        # Calculate total score with weights
        df["condition_score"] = sum(conditions.values())
        
        # Normalize score to 0-10 range
        df["condition_score"] = df["condition_score"] / 15 * 10
        
        # Add position sizing based on score
        df["position_size"] = np.where(
            df["condition_score"] > 7,
            1.0,
            df["condition_score"] / 7
        )

        condition = (
            macd_strict_condition
            & (~strong_bearish_candle)  # Don't buy on strong bearish candle
            & (curr_not_strong_bearish)  # Current candle shows strength
            & (prev_bearish)  # Previous candle was bearish (dump candle)
        )

        df.loc[condition, "buy"] = 1

        return df


    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
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
        Dynamic stoploss with drawdown protection:
        - Hard stop at -3%
        - ATR-based dynamic stoploss
        - Additional protection based on drawdown
        """
        # Check maximum drawdown
        if self._current_drawdown > self.max_drawdown:
            # Tighter stop in drawdown mode
            return -0.02  # 2% stop in drawdown
            
        # Base stoploss
        if current_profit < -0.03:  # Hard stop at 3%
            return -0.03

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
            
        current_atr = dataframe["atr_normalized"].iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            return self.stoploss
            
        # Adjust stop based on profit and market conditions
        if current_profit > 0.02:  # In profit
            dynamic_stop = -(current_atr * 1.2)  # Tighter stop
        else:
            dynamic_stop = -(current_atr * 1.8)  # Wider stop
            
        # Apply minimum stoploss
        return max(dynamic_stop, -0.03)

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
        Additional trade entry confirmation with volatility, position sizing, and drawdown protection
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return True
            
        # Get current market conditions
        current_atr = dataframe["atr_normalized"].iloc[-1]
        current_volatility = dataframe["volatility_filter"].iloc[-1]
        current_confluence = dataframe["confluence_score"].iloc[-1]
        current_market_regime = dataframe["market_regime"].iloc[-1]
        
        # Check drawdown protection
        if self._current_drawdown > self.max_drawdown:
            logger.info(f"Skipping {pair} due to maximum drawdown reached: {self._current_drawdown:.2%}")
            return False
            
        # Adjust volatility filter threshold based on market regime
        volatility_threshold = 0.02 if current_market_regime == 1 else 0.03
        
        # Only skip if volatility is extremely high
        if current_volatility > volatility_threshold:
            logger.info(f"Skipping {pair} due to extreme volatility: {current_volatility}")
            return False
            
        # Check confluence score
        if current_confluence < self.confluence_threshold * 0.9:
            logger.info(f"Skipping {pair} due to low confluence score: {current_confluence}")
            return False
            
        # Calculate position size based on multiple factors
        if current_atr > 0:
            # Base position size based on ATR
            atr_based_size = min(
                self.max_position_size,
                self.min_position_size + (1 / current_atr) * 0.02
            )
            
            # Adjust based on market conditions
            market_adjustment = 1.0
            if current_market_regime == 1:  # Trending market
                market_adjustment = 1.2
            elif current_market_regime == -1:  # Volatile market
                market_adjustment = 0.8
                
            # Final position size
            position_size = min(
                self.max_position_size,
                max(self.min_position_size, atr_based_size * market_adjustment)
            )
            
            logger.info(
                f"Adjusted position size: {position_size:.2%}"
                f" (ATR: {atr_based_size:.2%}, Market: {market_adjustment:.2f})"
            )
            
            # Update trade parameters
            kwargs["stake_amount"] = position_size * self.wallet_balance
            
        return True
        
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        """
        Custom stake amount calculation based on position sizing
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
            
        current_atr = dataframe["atr_normalized"].iloc[-1]
        if current_atr > 0:
            # Calculate position size based on ATR and market conditions
            position_size = min(
                self.max_position_size,
                max(self.min_position_size, (1 / current_atr) * 0.02)
            )
            
            return position_size * self.wallet_balance
            
        return proposed_stake

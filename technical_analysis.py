"""
Technical Analysis Module
Calculates various technical indicators, recognizes chart patterns, and generates trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# Try to import scipy for signal processing, use numpy fallback if not available
try:
    from scipy.signal import argrelextrema
except ImportError:
    # Use numpy-based fallback
    def argrelextrema(data, comparator, order=1, mode='clip'):
        """Fallback implementation of argrelextrema using numpy."""
        data = np.asarray(data)
        if len(data) < 2 * order + 1:
            return np.array([], dtype=int)

        extrema = []
        for i in range(order, len(data) - order):
            window = data[i-order:i+order+1]
            if len(window) > 0 and not np.any(np.isnan(window)):
                if comparator(data[i], window).all():
                    extrema.append(i)
        return np.array(extrema, dtype=int)


class TechnicalIndicators:
    """Calculate various technical indicators."""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return {
            'k': k_percent,
            'd': d_percent
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        high_max = high.rolling(window=period).max()
        low_min = low.rolling(window=period).min()
        return -100 * ((high_max - close) / (high_max - low_min))

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci


class ChartPatternRecognizer:
    """Recognize common chart patterns in price data."""

    def __init__(self, min_period: int = 5, max_period: int = 60):
        self.min_period = min_period
        self.max_period = max_period

    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect all chart patterns and return results."""
        if df.empty or len(df) < self.min_period * 2:
            return {'patterns': [], 'bullish_patterns': [], 'bearish_patterns': []}

        close = df['close']
        high = df['high']
        low = df['low']

        patterns = []

        # Detect various patterns with error handling
        try:
            patterns.extend(self._detect_double_top_bottom(close))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_head_shoulders(close, high, low))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_triple_top_bottom(close))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_flags_pennants(close, high, low))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_wedges(close, high, low))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_triangles(close, high, low))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_cup_handle(close))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_rounding_bottom(close))
        except Exception:
            pass
        try:
            patterns.extend(self._detect_support_resistance_breakout(close, high, low))
        except Exception:
            pass

        # Separate by sentiment
        bullish = [p for p in patterns if p.get('sentiment') == 'bullish']
        bearish = [p for p in patterns if p.get('sentiment') == 'bearish']

        return {
            'patterns': patterns,
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'pattern_count': len(patterns)
        }

    def _detect_double_top_bottom(self, close: pd.Series) -> List[Dict]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []

        try:
            window = max(5, len(close) // 20)

            # Find local maxima and minima
            maxima_result = argrelextrema(close.values, np.greater, order=window)
            maxima_idx = maxima_result[0] if len(maxima_result) > 0 and len(maxima_result[0]) > 0 else np.array([])

            minima_result = argrelextrema(close.values, np.less, order=window)
            minima_idx = minima_result[0] if len(minima_result) > 0 and len(minima_result[0]) > 0 else np.array([])

            # Double Top: Two similar peaks with a trough between
            if len(maxima_idx) >= 2:
                for i in range(len(maxima_idx) - 1):
                    idx1, idx2 = maxima_idx[i], maxima_idx[i + 1]
                    if idx2 - idx1 < self.max_period:
                        price1, price2 = close.iloc[idx1], close.iloc[idx2]
                        trough_price = close.iloc[idx1:idx2].min()

                        # Check if peaks are similar (within 3%)
                        if price1 > 0 and abs(price1 - price2) / price1 < 0.03:
                            # Check if there's a valley between
                            if trough_price < price1 * 0.97 and trough_price < price2 * 0.97:
                                current_price = close.iloc[-1]
                                # Bearish if we're near the peaks or breaking down
                                if current_price < price2 * 0.98:
                                    patterns.append({
                                        'name': 'Double Top',
                                        'sentiment': 'bearish',
                                        'reliability': 'medium',
                                        'description': f"Double Top formed at ${price1:.2f}, neckline at ${trough_price:.2f}",
                                        'signal_strength': -2
                                    })

            # Double Bottom: Two similar troughs with a peak between
            if len(minima_idx) >= 2:
                for i in range(len(minima_idx) - 1):
                    idx1, idx2 = minima_idx[i], minima_idx[i + 1]
                    if idx2 - idx1 < self.max_period:
                        price1, price2 = close.iloc[idx1], close.iloc[idx2]
                        peak_price = close.iloc[idx1:idx2].max()

                        # Check if troughs are similar (within 3%)
                        if price1 > 0 and abs(price1 - price2) / price1 < 0.03:
                            # Check if there's a peak between
                            if peak_price > price1 * 1.03 and peak_price > price2 * 1.03:
                                current_price = close.iloc[-1]
                                # Bullish if we're breaking above neckline
                                if current_price > peak_price * 0.98:
                                    patterns.append({
                                        'name': 'Double Bottom',
                                        'sentiment': 'bullish',
                                        'reliability': 'medium',
                                        'description': f"Double Bottom formed at ${price1:.2f}, neckline at ${peak_price:.2f}",
                                        'signal_strength': 2
                                    })
        except Exception:
            pass  # Pattern detection failed, return empty list

        return patterns

    def _detect_head_shoulders(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict]:
        """Detect Head and Shoulders and Inverse Head and Shoulders."""
        patterns = []
        window = max(5, len(close) // 15)

        # Find local maxima for H&S
        maxima_idx = argrelextrema(close.values, np.greater, order=window)[0]

        if len(maxima_idx) >= 3:
            for i in range(len(maxima_idx) - 2):
                idx1, idx2, idx3 = maxima_idx[i], maxima_idx[i + 1], maxima_idx[i + 2]
                if idx3 - idx1 < self.max_period * 2:
                    left_shoulder = close.iloc[idx1]
                    head = close.iloc[idx2]
                    right_shoulder = close.iloc[idx3]

                    # Head should be highest
                    if head > left_shoulder and head > right_shoulder:
                        # Shoulders should be roughly similar (within 5%)
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                            # Find neckline (lowest point between left shoulder and right shoulder)
                            neckline = close.iloc[idx1:idx3].min()
                            current_price = close.iloc[-1]

                            # Check if pattern is completing (price near or below neckline)
                            if current_price < neckline * 1.02:
                                patterns.append({
                                    'name': 'Head and Shoulders',
                                    'sentiment': 'bearish',
                                    'reliability': 'high',
                                    'description': f"H&S pattern with head at ${head:.2f}, neckline at ${neckline:.2f}",
                                    'signal_strength': -3
                                })

        # Inverse H&S (using minima)
        minima_idx = argrelextrema(close.values, np.less, order=window)[0]

        if len(minima_idx) >= 3:
            for i in range(len(minima_idx) - 2):
                idx1, idx2, idx3 = minima_idx[i], minima_idx[i + 1], minima_idx[i + 2]
                if idx3 - idx1 < self.max_period * 2:
                    left_shoulder = close.iloc[idx1]
                    head = close.iloc[idx2]
                    right_shoulder = close.iloc[idx3]

                    # Head should be lowest
                    if head < left_shoulder and head < right_shoulder:
                        # Shoulders should be roughly similar
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                            # Find neckline (highest point between)
                            neckline = close.iloc[idx1:idx3].max()
                            current_price = close.iloc[-1]

                            # Check if breaking above neckline
                            if current_price > neckline * 0.98:
                                patterns.append({
                                    'name': 'Inverse Head and Shoulders',
                                    'sentiment': 'bullish',
                                    'reliability': 'high',
                                    'description': f"Inverse H&S with head at ${head:.2f}, neckline at ${neckline:.2f}",
                                    'signal_strength': 3
                                })

        return patterns

    def _detect_triple_top_bottom(self, close: pd.Series) -> List[Dict]:
        """Detect Triple Top and Triple Bottom patterns."""
        patterns = []
        window = max(5, len(close) // 15)

        maxima_idx = argrelextrema(close.values, np.greater, order=window)[0]
        minima_idx = argrelextrema(close.values, np.less, order=window)[0]

        # Triple Top
        if len(maxima_idx) >= 3:
            for i in range(len(maxima_idx) - 2):
                idx1, idx2, idx3 = maxima_idx[i], maxima_idx[i + 1], maxima_idx[i + 2]
                if idx3 - idx1 < self.max_period * 2:
                    prices = [close.iloc[idx1], close.iloc[idx2], close.iloc[idx3]]
                    avg_price = np.mean(prices)

                    # All three peaks should be similar (within 3%)
                    if all(abs(p - avg_price) / avg_price < 0.03 for p in prices):
                        neckline = close.iloc[idx1:idx3].min()
                        current_price = close.iloc[-1]

                        if current_price < neckline * 1.02:
                            patterns.append({
                                'name': 'Triple Top',
                                'sentiment': 'bearish',
                                'reliability': 'high',
                                'description': f"Triple Top at ${avg_price:.2f}, neckline at ${neckline:.2f}",
                                'signal_strength': -3
                            })

        # Triple Bottom
        if len(minima_idx) >= 3:
            for i in range(len(minima_idx) - 2):
                idx1, idx2, idx3 = minima_idx[i], minima_idx[i + 1], minima_idx[i + 2]
                if idx3 - idx1 < self.max_period * 2:
                    prices = [close.iloc[idx1], close.iloc[idx2], close.iloc[idx3]]
                    avg_price = np.mean(prices)

                    # All three troughs should be similar
                    if all(abs(p - avg_price) / avg_price < 0.03 for p in prices):
                        neckline = close.iloc[idx1:idx3].max()
                        current_price = close.iloc[-1]

                        if current_price > neckline * 0.98:
                            patterns.append({
                                'name': 'Triple Bottom',
                                'sentiment': 'bullish',
                                'reliability': 'high',
                                'description': f"Triple Bottom at ${avg_price:.2f}, neckline at ${neckline:.2f}",
                                'signal_strength': 3
                            })

        return patterns

    def _detect_flags_pennants(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict]:
        """Detect Bullish/Bearish Flags and Pennants."""
        patterns = []
        if len(close) < 20:
            return patterns

        # Calculate trend
        recent_prices = close.tail(20)
        initial_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # Check for consolidation (narrowing range)
        recent_highs = high.tail(10)
        recent_lows = low.tail(10)
        avg_range = (recent_highs.max() - recent_lows.min()) / recent_lows.min()

        # Bullish Flag: Strong uptrend followed by consolidation
        if initial_trend > 0.05:
            # Check for consolidation (price moving sideways)
            consolidation_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            if abs(consolidation_trend) < 0.02:
                patterns.append({
                    'name': 'Bullish Flag',
                    'sentiment': 'bullish',
                    'reliability': 'medium',
                    'description': f"Bullish flag pattern after {initial_trend*100:.1f}% uptrend, consolidating before continuation",
                    'signal_strength': 2
                })

        # Bearish Flag: Strong downtrend followed by consolidation
        elif initial_trend < -0.05:
            consolidation_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            if abs(consolidation_trend) < 0.02:
                patterns.append({
                    'name': 'Bearish Flag',
                    'sentiment': 'bearish',
                    'reliability': 'medium',
                    'description': f"Bearish flag pattern after {initial_trend*100:.1f}% downtrend, consolidating before continuation",
                    'signal_strength': -2
                })

        # Pennants: Converging highs and lows
        highs_slope = self._calculate_slope(recent_highs.values[-5:])
        lows_slope = self._calculate_slope(recent_lows.values[-5:])

        # Bullish Pennant: Highs descending, lows ascending (converging)
        if highs_slope < 0 and lows_slope > 0 and initial_trend > 0.03:
            patterns.append({
                'name': 'Bullish Pennant',
                'sentiment': 'bullish',
                'reliability': 'medium',
                'description': "Bullish pennant - converging price action suggests continuation of uptrend",
                'signal_strength': 2
            })

        # Bearish Pennant
        if highs_slope < 0 and lows_slope > 0 and initial_trend < -0.03:
            patterns.append({
                'name': 'Bearish Pennant',
                'sentiment': 'bearish',
                'reliability': 'medium',
                'description': "Bearish pennant - converging price action suggests continuation of downtrend",
                'signal_strength': -2
            })

        return patterns

    def _detect_wedges(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict]:
        """Detect Rising Wedge and Falling Wedge patterns."""
        patterns = []
        if len(close) < 20:
            return patterns

        recent_highs = high.tail(15).values
        recent_lows = low.tail(15).values

        highs_slope = self._calculate_slope(recent_highs)
        lows_slope = self._calculate_slope(recent_lows)

        # Rising Wedge: Both highs and lows rising, but lows rising faster (bearish)
        if highs_slope > 0 and lows_slope > 0 and lows_slope > highs_slope * 1.5:
            patterns.append({
                'name': 'Rising Wedge',
                'sentiment': 'bearish',
                'reliability': 'medium',
                'description': "Rising wedge pattern - typically leads to downside break",
                'signal_strength': -1.5
            })

        # Falling Wedge: Both highs and lows falling, but highs falling faster (bullish)
        if highs_slope < 0 and lows_slope < 0 and abs(highs_slope) > abs(lows_slope) * 1.5:
            patterns.append({
                'name': 'Falling Wedge',
                'sentiment': 'bullish',
                'reliability': 'medium',
                'description': "Falling wedge pattern - typically leads to upside break",
                'signal_strength': 1.5
            })

        return patterns

    def _detect_triangles(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict]:
        """Detect Ascending, Descending, and Symmetrical Triangles."""
        patterns = []
        if len(close) < 20:
            return patterns

        recent_highs = high.tail(20).values
        recent_lows = low.tail(20).values

        # Calculate slopes
        highs_slope = self._calculate_slope(recent_highs[-10:])
        lows_slope = self._calculate_slope(recent_lows[-10:])

        current_price = close.iloc[-1]
        price_range = close.tail(20).max() - close.tail(20).min()

        # Ascending Triangle: Flat top (highs stable), rising lows
        if abs(highs_slope) < 0.001 and lows_slope > 0:
            resistance = recent_highs[-10:].max()
            if current_price > resistance * 0.98:
                patterns.append({
                    'name': 'Ascending Triangle',
                    'sentiment': 'bullish',
                    'reliability': 'high',
                    'description': f"Ascending triangle with resistance at ${resistance:.2f} - bullish breakout potential",
                    'signal_strength': 2
                })

        # Descending Triangle: Flat bottom (lows stable), falling highs
        if abs(lows_slope) < 0.001 and highs_slope < 0:
            support = recent_lows[-10:].min()
            if current_price < support * 1.02:
                patterns.append({
                    'name': 'Descending Triangle',
                    'sentiment': 'bearish',
                    'reliability': 'high',
                    'description': f"Descending triangle with support at ${support:.2f} - bearish breakdown risk",
                    'signal_strength': -2
                })

        # Symmetrical Triangle: Converging highs and lows
        if highs_slope < 0 and lows_slope > 0:
            # Check if converging
            range_start = (recent_highs[0] - recent_lows[0])
            range_end = (recent_highs[-1] - recent_lows[-1])
            if range_end < range_start * 0.7:  # Range narrowed by at least 30%
                patterns.append({
                    'name': 'Symmetrical Triangle',
                    'sentiment': 'neutral',
                    'reliability': 'low',
                    'description': "Symmetrical triangle consolidation - breakout direction determines next move",
                    'signal_strength': 0
                })

        return patterns

    def _detect_cup_handle(self, close: pd.Series) -> List[Dict]:
        """Detect Cup and Handle pattern."""
        patterns = []
        if len(close) < 40:
            return patterns

        # Look for cup shape (rounded bottom) in recent data
        cup_data = close.tail(40)
        min_idx = cup_data.argmin()
        min_price = cup_data.iloc[min_idx]

        # Cup should have rounded bottom (gradual decline and rise)
        left_peak = cup_data.iloc[:min_idx].max()
        right_peak = cup_data.iloc[min_idx:].max()

        # Both sides should be higher than the bottom
        if left_peak > min_price * 1.1 and right_peak > min_price * 1.1:
            # Check for handle (small pullback after right peak)
            handle_data = close.tail(10)
            handle_decline = (handle_data.max() - handle_data.min()) / handle_data.max()

            # Handle should be small (less than 10% decline)
            if handle_decline < 0.1 and handle_decline > 0.02:
                current_price = close.iloc[-1]
                # Bullish if breaking above handle
                if current_price > handle_data.max() * 0.98:
                    patterns.append({
                        'name': 'Cup and Handle',
                        'sentiment': 'bullish',
                        'reliability': 'high',
                        'description': f"Cup and handle pattern with cup bottom at ${min_price:.2f}",
                        'signal_strength': 3
                    })

        return patterns

    def _detect_rounding_bottom(self, close: pd.Series) -> List[Dict]:
        """Detect Rounding Bottom pattern."""
        patterns = []
        if len(close) < 30:
            return patterns

        data = close.tail(30)
        min_idx = data.argmin()
        min_val = data.iloc[min_idx]

        # Check for U-shape: gradual decline to minimum, gradual rise
        decline_slope = self._calculate_slope(data.iloc[:min_idx+1].values)
        rise_slope = self._calculate_slope(data.iloc[min_idx:].values)

        # Should have negative decline (going down) and positive rise (going up)
        if decline_slope < 0 and rise_slope > 0:
            # Current price should be near the high of the pattern
            current_price = close.iloc[-1]
            pattern_high = data.max()

            if current_price > pattern_high * 0.95:
                patterns.append({
                    'name': 'Rounding Bottom',
                    'sentiment': 'bullish',
                    'reliability': 'medium',
                    'description': f"Rounding bottom pattern formed at ${min_val:.2f}",
                    'signal_strength': 2
                })

        return patterns

    def _detect_support_resistance_breakout(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict]:
        """Detect support/resistance breakouts."""
        patterns = []

        # Find key resistance levels (recent highs cluster)
        recent_highs = high.tail(30)
        resistance_levels = self._find_price_levels(recent_highs, level_type='resistance')

        # Find key support levels (recent lows cluster)
        recent_lows = low.tail(30)
        support_levels = self._find_price_levels(recent_lows, level_type='support')

        current_price = close.iloc[-1]

        # Check for bullish breakout above resistance
        for level in resistance_levels:
            if current_price > level * 1.01 and current_price < level * 1.05:
                patterns.append({
                    'name': 'Resistance Breakout',
                    'sentiment': 'bullish',
                    'reliability': 'medium',
                    'description': f"Breaking above resistance at ${level:.2f}",
                    'signal_strength': 1.5
                })
                break

        # Check for bearish breakdown below support
        for level in support_levels:
            if current_price < level * 0.99 and current_price > level * 0.95:
                patterns.append({
                    'name': 'Support Breakdown',
                    'sentiment': 'bearish',
                    'reliability': 'medium',
                    'description': f"Breaking below support at ${level:.2f}",
                    'signal_strength': -1.5
                })
                break

        return patterns

    def _calculate_slope(self, data: np.array) -> float:
        """Calculate the linear slope of data."""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        try:
            slope = np.polyfit(x, data, 1)[0]
            # Normalize slope by the data range
            normalized_slope = slope / (np.mean(data) if np.mean(data) != 0 else 1)
            return normalized_slope
        except:
            return 0

    def _find_price_levels(self, price_series: pd.Series, level_type: str = 'resistance') -> List[float]:
        """Find significant price levels where price clustered."""
        levels = []
        if len(price_series) < 5:
            return levels

        prices = price_series.dropna().values
        if len(prices) < 5:
            return levels

        # Cluster prices into levels
        price_range = prices.max() - prices.min()
        if price_range == 0:
            return levels

        cluster_size = price_range * 0.02  # 2% of range for clustering

        for i, price in enumerate(prices):
            # Check if this price is near other prices (cluster)
            nearby = prices[np.abs(prices - price) < cluster_size]
            if len(nearby) >= 2:  # At least 2 touches
                avg_level = np.mean(nearby)
                if all(abs(avg_level - l) > cluster_size for l in levels):
                    levels.append(avg_level)

        # Sort by significance (number of touches)
        levels.sort(key=lambda x: -len([p for p in prices if abs(p - x) < cluster_size]))
        return levels[:3]  # Return top 3 levels


class SignalAnalyzer:
    """Analyze indicators, chart patterns, and generate trading signals."""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.pattern_recognizer = ChartPatternRecognizer()

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform complete technical analysis on price data including pattern recognition."""
        if df.empty or len(df) < 50:
            return {'error': 'Insufficient data for analysis'}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else None

        # Calculate all indicators
        results = {
            'price': close.iloc[-1],
            'price_change_pct': ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100,
        }

        # Moving Averages
        results['sma_20'] = self.indicators.sma(close, 20).iloc[-1]
        results['sma_50'] = self.indicators.sma(close, 50).iloc[-1]
        results['sma_200'] = self.indicators.sma(close, 200).iloc[-1]
        results['ema_12'] = self.indicators.ema(close, 12).iloc[-1]
        results['ema_26'] = self.indicators.ema(close, 26).iloc[-1]

        # RSI
        rsi_values = self.indicators.rsi(close, 14)
        results['rsi'] = rsi_values.iloc[-1]
        results['rsi_prev'] = rsi_values.iloc[-2]

        # MACD
        macd_data = self.indicators.macd(close)
        results['macd'] = macd_data['macd'].iloc[-1]
        results['macd_signal'] = macd_data['signal'].iloc[-1]
        results['macd_histogram'] = macd_data['histogram'].iloc[-1]
        results['macd_histogram_prev'] = macd_data['histogram'].iloc[-2]

        # Bollinger Bands
        bb = self.indicators.bollinger_bands(close)
        results['bb_upper'] = bb['upper'].iloc[-1]
        results['bb_middle'] = bb['middle'].iloc[-1]
        results['bb_lower'] = bb['lower'].iloc[-1]
        results['bb_width'] = ((bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / bb['middle'].iloc[-1]) * 100

        # Stochastic
        stoch = self.indicators.stochastic(high, low, close)
        results['stoch_k'] = stoch['k'].iloc[-1]
        results['stoch_d'] = stoch['d'].iloc[-1]

        # ATR
        results['atr'] = self.indicators.atr(high, low, close).iloc[-1]
        results['atr_pct'] = (results['atr'] / close.iloc[-1]) * 100

        # Williams %R
        results['williams_r'] = self.indicators.williams_r(high, low, close).iloc[-1]

        # CCI
        results['cci'] = self.indicators.cci(high, low, close).iloc[-1]

        # Volume analysis (if available)
        if volume is not None and not volume.isna().all():
            results['volume'] = volume.iloc[-1]
            results['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
            results['volume_ratio'] = results['volume'] / results['volume_sma_20'] if results['volume_sma_20'] > 0 else 1

        # Support and Resistance (recent highs/lows)
        results['recent_high_52w'] = high.rolling(252).max().iloc[-1] if len(high) >= 252 else high.max()
        results['recent_low_52w'] = low.rolling(252).min().iloc[-1] if len(low) >= 252 else low.min()

        # Detect chart patterns
        patterns = self.pattern_recognizer.detect_all_patterns(df)
        results['patterns'] = patterns.get('patterns', [])
        results['bullish_patterns'] = patterns.get('bullish_patterns', [])
        results['bearish_patterns'] = patterns.get('bearish_patterns', [])

        return results

    def generate_signals(self, analysis: Dict, df: pd.DataFrame = None) -> Dict:
        """Generate buy/sell signals based on indicator analysis and chart patterns.
        Uses more conservative thresholds to avoid overly optimistic signals.
        Tracks points for each factor for transparency.
        """
        if 'error' in analysis:
            return {'signal': 'N/A', 'strength': 0, 'reason': analysis['error']}

        signals = []
        confidence = 0
        bullish_factors = []
        bearish_factors = []
        point_breakdown = []  # Track each factor with its points

        price = analysis['price']
        price_change_pct = analysis.get('price_change_pct', 0)

        # Chart Pattern Analysis
        bullish_patterns = analysis.get('bullish_patterns', [])
        bearish_patterns = analysis.get('bearish_patterns', [])
        all_patterns = analysis.get('patterns', [])

        # Add bullish pattern signals
        for pattern in bullish_patterns:
            pattern_name = pattern.get('name', 'Unknown')
            reliability = pattern.get('reliability', 'medium')
            strength = pattern.get('signal_strength', 1)
            description = pattern.get('description', '')

            signals.append(f"Chart Pattern: {pattern_name} detected")
            bullish_factors.append(f"{pattern_name} pattern ({reliability} reliability): {description}")
            point_breakdown.append({
                'factor': f"{pattern_name} Pattern",
                'points': strength,
                'category': 'pattern',
                'sentiment': 'bullish'
            })
            confidence += strength

        # Add bearish pattern signals
        for pattern in bearish_patterns:
            pattern_name = pattern.get('name', 'Unknown')
            reliability = pattern.get('reliability', 'medium')
            strength = pattern.get('signal_strength', -1)
            description = pattern.get('description', '')

            signals.append(f"Chart Pattern: {pattern_name} detected")
            bearish_factors.append(f"{pattern_name} pattern ({reliability} reliability): {description}")
            point_breakdown.append({
                'factor': f"{pattern_name} Pattern",
                'points': strength,
                'category': 'pattern',
                'sentiment': 'bearish'
            })
            confidence += strength  # strength is negative for bearish patterns

        # RSI Signals - More conservative thresholds
        rsi = analysis.get('rsi', 50)
        if rsi < 25:  # More extreme oversold needed
            points = 1.5  # Reduced from 2
            signals.append(f"RSI ({rsi:.1f}) deeply oversold - BUY signal")
            bullish_factors.append(f"RSI at {rsi:.1f} is deeply oversold (<25), suggesting potential bounce opportunity")
            point_breakdown.append({
                'factor': f'RSI ({rsi:.1f})',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bullish',
                'detail': 'Deeply oversold (<25)'
            })
            confidence += points
        elif rsi > 75:  # Lower overbought threshold
            points = -1.5  # Reduced from -2
            signals.append(f"RSI ({rsi:.1f}) overbought - SELL signal")
            bearish_factors.append(f"RSI at {rsi:.1f} is overbought (>75), suggesting potential pullback")
            point_breakdown.append({
                'factor': f'RSI ({rsi:.1f})',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bearish',
                'detail': 'Overbought (>75)'
            })
            confidence += points
        elif rsi < 35:
            points = 0.5  # Reduced from 1
            signals.append(f"RSI ({rsi:.1f}) approaching oversold territory")
            bullish_factors.append(f"RSI at {rsi:.1f} is approaching oversold levels (<35)")
            point_breakdown.append({
                'factor': f'RSI ({rsi:.1f})',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bullish',
                'detail': 'Approaching oversold (<35)'
            })
            confidence += points
        elif rsi > 65:
            points = -0.5  # Reduced from -1
            signals.append(f"RSI ({rsi:.1f}) approaching overbought territory")
            bearish_factors.append(f"RSI at {rsi:.1f} is approaching overbought levels (>65)")
            point_breakdown.append({
                'factor': f'RSI ({rsi:.1f})',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bearish',
                'detail': 'Approaching overbought (>65)'
            })
            confidence += points

        # MACD Signals - Requires stronger confirmation
        macd_hist = analysis.get('macd_histogram', 0)
        macd_hist_prev = analysis.get('macd_histogram_prev', 0)
        macd = analysis.get('macd', 0)
        macd_signal = analysis.get('macd_signal', 0)

        # Only count bullish MACD if histogram is positive AND MACD > signal
        if macd_hist > 0 and macd_hist_prev <= 0 and macd > macd_signal:
            points = 1.5  # Reduced from 2
            signals.append("MACD bullish crossover - BUY signal")
            bullish_factors.append(f"MACD histogram crossed above zero with MACD ({macd:.2f}) > signal ({macd_signal:.2f})")
            point_breakdown.append({
                'factor': 'MACD Crossover',
                'points': points,
                'category': 'trend',
                'sentiment': 'bullish',
                'detail': f'Histogram +{macd_hist:.2f}, MACD > Signal'
            })
            confidence += points
        elif macd_hist < 0 and macd_hist_prev >= 0 and macd < macd_signal:
            points = -1.5  # Reduced from -2
            signals.append("MACD bearish crossover - SELL signal")
            bearish_factors.append(f"MACD histogram crossed below zero with MACD ({macd:.2f}) < signal ({macd_signal:.2f})")
            point_breakdown.append({
                'factor': 'MACD Crossover',
                'points': points,
                'category': 'trend',
                'sentiment': 'bearish',
                'detail': f'Histogram {macd_hist:.2f}, MACD < Signal'
            })
            confidence += points
        elif macd > macd_signal:
            # Mild bullish - only adds small confidence
            if macd_hist > 0:
                points = 0.25  # Reduced from 0.5
                bullish_factors.append(f"MACD ({macd:.2f}) is above signal line with positive histogram")
                point_breakdown.append({
                    'factor': 'MACD Position',
                    'points': points,
                    'category': 'trend',
                    'sentiment': 'bullish',
                    'detail': 'Above signal line, positive histogram'
                })
                confidence += points
        elif macd < macd_signal:
            # Mild bearish
            if macd_hist < 0:
                points = -0.25  # Reduced from -0.5
                bearish_factors.append(f"MACD ({macd:.2f}) is below signal line with negative histogram")
                point_breakdown.append({
                    'factor': 'MACD Position',
                    'points': points,
                    'category': 'trend',
                    'sentiment': 'bearish',
                    'detail': 'Below signal line, negative histogram'
                })
                confidence += points

        # Moving Average Crossover - Conservative approach
        sma_20 = analysis.get('sma_20', price)
        sma_50 = analysis.get('sma_50', price)
        sma_200 = analysis.get('sma_200', price)

        # Golden cross requires price above all MAs
        if price > sma_20 > sma_50 > sma_200 * 0.98:
            points = 1  # Reduced from 1.5
            signals.append("Price above MA(20) > MA(50) with bullish alignment")
            bullish_factors.append(f"Price is above 20-day and 50-day MAs with bullish alignment")
            point_breakdown.append({
                'factor': 'Moving Average Alignment',
                'points': points,
                'category': 'trend',
                'sentiment': 'bullish',
                'detail': 'Price > MA20 > MA50 (Golden Cross setup)'
            })
            confidence += points
        elif price < sma_20 < sma_50:
            points = -1  # Reduced from -1.5
            signals.append("Price below MA(20) < MA(50) - bearish trend")
            bearish_factors.append(f"Price is below both 20-day and 50-day moving averages - downtrend")
            point_breakdown.append({
                'factor': 'Moving Average Alignment',
                'points': points,
                'category': 'trend',
                'sentiment': 'bearish',
                'detail': 'Price < MA20 < MA50 (Death Cross setup)'
            })
            confidence += points

        # 200-day MA is crucial for long-term trend
        if price > sma_200:
            bullish_factors.append(f"Price is above the 200-day moving average - long-term uptrend")
            point_breakdown.append({
                'factor': '200-day MA',
                'points': 0,
                'category': 'trend',
                'sentiment': 'bullish',
                'detail': 'Above 200-day MA (no points - trend factor only)'
            })
        elif price < sma_200:
            points = -0.5  # Reduced from -1
            bearish_factors.append(f"Price is below the 200-day moving average - long-term downtrend")
            point_breakdown.append({
                'factor': '200-day MA',
                'points': points,
                'category': 'trend',
                'sentiment': 'bearish',
                'detail': 'Below 200-day MA'
            })
            confidence += points

        # Bollinger Bands - More bearish when above upper band
        bb_upper = analysis.get('bb_upper', price * 1.1)
        bb_lower = analysis.get('bb_lower', price * 0.9)
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        if price > bb_upper:
            points = -1  # Reduced from -1.5
            signals.append(f"Price above upper Bollinger Band - overbought")
            bearish_factors.append(f"Price is above the upper Bollinger Band - overextended and likely to pull back")
            point_breakdown.append({
                'factor': 'Bollinger Bands',
                'points': points,
                'category': 'volatility',
                'sentiment': 'bearish',
                'detail': f'Above upper band ({bb_position:.1%} position)'
            })
            confidence += points
        elif price < bb_lower:
            points = 0.75  # Reduced from 1
            signals.append(f"Price below lower Bollinger Band - oversold")
            bullish_factors.append(f"Price is below the lower Bollinger Band - oversold conditions")
            point_breakdown.append({
                'factor': 'Bollinger Bands',
                'points': points,
                'category': 'volatility',
                'sentiment': 'bullish',
                'detail': f'Below lower band ({bb_position:.1%} position)'
            })
            confidence += points

        # Stochastic - More conservative
        stoch_k = analysis.get('stoch_k', 50)
        stoch_d = analysis.get('stoch_d', 50)

        if stoch_k < 15 and stoch_d < 15:
            points = 1  # Reduced from 1.5
            signals.append(f"Stochastic ({stoch_k:.1f}/{stoch_d:.1f}) deeply oversold - potential BUY")
            bullish_factors.append(f"Stochastic is deeply oversold (<15), suggesting potential bounce")
            point_breakdown.append({
                'factor': 'Stochastic',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bullish',
                'detail': f'{stoch_k:.1f}/{stoch_d:.1f} - Deeply oversold'
            })
            confidence += points
        elif stoch_k > 85 and stoch_d > 85:
            points = -1  # Reduced from -1.5
            signals.append(f"Stochastic ({stoch_k:.1f}/{stoch_d:.1f}) overbought - potential SELL")
            bearish_factors.append(f"Stochastic is overbought (>85), suggesting pullback risk")
            point_breakdown.append({
                'factor': 'Stochastic',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bearish',
                'detail': f'{stoch_k:.1f}/{stoch_d:.1f} - Overbought'
            })
            confidence += points

        # Williams %R - More extreme thresholds
        williams_r = analysis.get('williams_r', -50)
        if williams_r < -85:
            points = 1
            bullish_factors.append(f"Williams %R at {williams_r:.1f} indicates deeply oversold conditions")
            point_breakdown.append({
                'factor': "Williams %R",
                'points': points,
                'category': 'momentum',
                'sentiment': 'bullish',
                'detail': f'{williams_r:.1f} - Oversold'
            })
            confidence += points
        elif williams_r > -15:
            points = -1
            bearish_factors.append(f"Williams %R at {williams_r:.1f} indicates overbought conditions")
            point_breakdown.append({
                'factor': "Williams %R",
                'points': points,
                'category': 'momentum',
                'sentiment': 'bearish',
                'detail': f'{williams_r:.1f} - Overbought'
            })
            confidence += points

        # CCI
        cci = analysis.get('cci', 0)
        if cci < -150:
            points = 1
            bullish_factors.append(f"CCI at {cci:.1f} is deeply oversold (< -150)")
            point_breakdown.append({
                'factor': 'CCI',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bullish',
                'detail': f'{cci:.1f} - Oversold'
            })
            confidence += points
        elif cci > 150:
            points = -1
            bearish_factors.append(f"CCI at {cci:.1f} is overbought (> 150)")
            point_breakdown.append({
                'factor': 'CCI',
                'points': points,
                'category': 'momentum',
                'sentiment': 'bearish',
                'detail': f'{cci:.1f} - Overbought'
            })
            confidence += points

        # Recent price momentum - Add bearish weight if down significantly
        if price_change_pct < -3:
            points = -1
            bearish_factors.append(f"Recent price decline of {price_change_pct:.1f}% shows selling pressure")
            point_breakdown.append({
                'factor': 'Price Momentum',
                'points': points,
                'category': 'price',
                'sentiment': 'bearish',
                'detail': f'{price_change_pct:+.1f}% - Selling pressure'
            })
            confidence += points
        elif price_change_pct > 3:
            points = 0.5
            bullish_factors.append(f"Recent price gain of {price_change_pct:.1f}% shows positive momentum")
            point_breakdown.append({
                'factor': 'Price Momentum',
                'points': points,
                'category': 'price',
                'sentiment': 'bullish',
                'detail': f'{price_change_pct:+.1f}% - Positive momentum'
            })
            confidence += points

        # Volume confirmation (if available)
        volume_multiplier_applied = False
        if 'volume_ratio' in analysis:
            vol_ratio = analysis['volume_ratio']
            if vol_ratio > 1.8:  # Higher threshold for volume confirmation
                old_confidence = confidence
                confidence = confidence * 1.1 if confidence > 0 else confidence
                volume_boost = confidence - old_confidence
                bullish_factors.append(f"Volume is {vol_ratio:.1f}x above average - strong conviction")
                point_breakdown.append({
                    'factor': 'Volume Confirmation',
                    'points': f'×1.1 (+{volume_boost:.2f})',
                    'category': 'volume',
                    'sentiment': 'bullish',
                    'detail': f'{vol_ratio:.1f}× average - Multiplier applied',
                    'is_multiplier': True
                })
                volume_multiplier_applied = True
            elif vol_ratio < 0.3:
                bearish_factors.append(f"Very low volume ({vol_ratio:.1f}x average) suggests weak participation")
                point_breakdown.append({
                    'factor': 'Volume',
                    'points': 0,
                    'category': 'volume',
                    'sentiment': 'bearish',
                    'detail': f'{vol_ratio:.1f}× average - Weak participation (no points)'
                })

        # 52-week position check - more conservative
        recent_high = analysis.get('recent_high_52w', price)
        recent_low = analysis.get('recent_low_52w', price)
        if recent_high > recent_low:
            price_position = (price - recent_low) / (recent_high - recent_low)
            if price_position > 0.9:
                points = -0.5
                bearish_factors.append(f"Price is in top 10% of 52-week range - extended")
                point_breakdown.append({
                    'factor': '52-Week Position',
                    'points': points,
                    'category': 'price',
                    'sentiment': 'bearish',
                    'detail': f'Top 10% of range - Extended'
                })
                confidence += points
            elif price_position < 0.2:
                points = 0.5
                bullish_factors.append(f"Price is in bottom 20% of 52-week range - value zone")
                point_breakdown.append({
                    'factor': '52-Week Position',
                    'points': points,
                    'category': 'price',
                    'sentiment': 'bullish',
                    'detail': f'Bottom 20% of range - Value zone'
                })
                confidence += points

        # Calculate base confidence (before volume multiplier) for display
        base_confidence = confidence
        if volume_multiplier_applied and confidence > 0:
            # Revert multiplier to show base
            base_confidence = confidence / 1.1

        # Determine overall signal - More conservative thresholds
        if confidence >= 8:
            overall = 'STRONG BUY'
            explanation = self._generate_signal_explanation('STRONG BUY', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        elif confidence >= 4:
            overall = 'BUY'
            explanation = self._generate_signal_explanation('BUY', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        elif confidence >= 1.5:
            overall = 'WEAK BUY'
            explanation = self._generate_signal_explanation('WEAK BUY', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        elif confidence <= -8:
            overall = 'STRONG SELL'
            explanation = self._generate_signal_explanation('STRONG SELL', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        elif confidence <= -4:
            overall = 'SELL'
            explanation = self._generate_signal_explanation('SELL', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        elif confidence <= -1.5:
            overall = 'WEAK SELL'
            explanation = self._generate_signal_explanation('WEAK SELL', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)
        else:
            overall = 'HOLD/NEUTRAL'
            explanation = self._generate_signal_explanation('HOLD', bullish_factors, bearish_factors, analysis, point_breakdown, base_confidence, confidence)

        return {
            'signal': overall,
            'confidence': confidence,
            'base_confidence': base_confidence,
            'reasons': signals,
            'explanation': explanation,
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'point_breakdown': point_breakdown
        }

    def _generate_signal_explanation(self, signal: str, bullish: list, bearish: list, analysis: Dict,
                                     point_breakdown: list = None, base_confidence: float = None,
                                     final_confidence: float = None) -> str:
        """Generate a detailed explanation of why the signal was generated."""
        price = analysis.get('price', 0)
        rsi = analysis.get('rsi', 50)
        change_pct = analysis.get('price_change_pct', 0)

        # Build point breakdown table
        points_table = ""
        if point_breakdown:
            points_table = "\n**Point Breakdown:**\n"

            # Group by category for better readability
            categories = {
                'pattern': 'Chart Patterns',
                'trend': 'Trend Indicators',
                'momentum': 'Momentum Indicators',
                'volatility': 'Volatility',
                'price': 'Price Action',
                'volume': 'Volume'
            }

            for cat_key, cat_name in categories.items():
                cat_items = [p for p in point_breakdown if p.get('category') == cat_key]
                if cat_items:
                    points_table += f"\n  {cat_name}:\n"
                    for item in cat_items:
                        pts = item['points']
                        detail = item.get('detail', '')
                        if isinstance(pts, str) or item.get('is_multiplier'):
                            points_table += f"    • {item['factor']}: {pts} - {detail}\n"
                        else:
                            sign = "+" if pts > 0 else ""
                            sentiment_icon = "📈" if item.get('sentiment') == 'bullish' else "📉" if item.get('sentiment') == 'bearish' else "➡"
                            points_table += f"    • {sentiment_icon} {item['factor']}: {sign}{pts} - {detail}\n"

            # Add total
            points_table += f"\n  **Base Score: {base_confidence if base_confidence is not None else final_confidence:.2f}**"
            if base_confidence is not None and final_confidence is not None and abs(final_confidence - base_confidence) > 0.1:
                points_table += f" → **Final Score: {final_confidence:.2f}** (after volume multiplier)"

        if signal == 'STRONG BUY':
            base = (f"**Strong Buy Signal** - Multiple technical indicators align to suggest a significant buying opportunity. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Key Bullish Factors:**\n"
            for factor in bullish[:5]:
                base += f"  • {factor}\n"
            if bearish:
                base += "\n**Cautionary Notes:**\n"
                for factor in bearish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Consider building positions with proper risk management. The confluence of oversold conditions and bullish momentum suggests potential upside."

        elif signal == 'BUY':
            base = (f"**Buy Signal** - Technical indicators favor the bullish side. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Supporting Factors:**\n"
            for factor in bullish[:4]:
                base += f"  • {factor}\n"
            if bearish:
                base += "\n**Watch For:**\n"
                for factor in bearish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Favorable entry point for long positions. Look for confirmation from volume continuation."

        elif signal == 'WEAK BUY':
            base = (f"**Weak Buy Signal** - Slight bullish bias but conviction is moderate. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Positive Indicators:**\n"
            for factor in bullish[:3]:
                base += f"  • {factor}\n"
            else:
                base += "Limited bullish confirmation available.\n"
            if bearish:
                base += "\n**Concerns:**\n"
                for factor in bearish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Approach with caution. Consider smaller position sizes or wait for more confirmation."

        elif signal == 'STRONG SELL':
            base = (f"**Strong Sell Signal** - Multiple indicators suggest significant downside risk. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Key Bearish Factors:**\n"
            for factor in bearish[:5]:
                base += f"  • {factor}\n"
            if bullish:
                base += "\n**Counter-arguments:**\n"
                for factor in bullish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Consider reducing exposure or taking profits. Overbought conditions and bearish momentum suggest potential decline."

        elif signal == 'SELL':
            base = (f"**Sell Signal** - Technical indicators favor the bearish side. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Concerning Factors:**\n"
            for factor in bearish[:4]:
                base += f"  • {factor}\n"
            if bullish:
                base += "\n**Supporting Elements:**\n"
                for factor in bullish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Favorable time to exit long positions or consider short opportunities with proper risk management."

        elif signal == 'WEAK SELL':
            base = (f"**Weak Sell Signal** - Slight bearish bias but conviction is moderate. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Negative Indicators:**\n"
            if bearish:
                for factor in bearish[:3]:
                    base += f"  • {factor}\n"
            else:
                base += "Limited bearish confirmation available.\n"
            if bullish:
                base += "\n**Positive Notes:**\n"
                for factor in bullish[:2]:
                    base += f"  • {factor}\n"
            base += "\n**Action:** Consider trimming positions or tightening stop losses. Wait for clearer signals before making major moves."

        else:  # HOLD/NEUTRAL
            base = (f"**Hold/Neutral Signal** - Technical indicators are mixed, showing no clear directional bias. "
                   f"Current price: ${price:.2f} ({change_pct:+.2f}%)")
            base += points_table
            base += "\n\n**Bullish Points:**\n" if bullish else ""
            for factor in bullish[:2]:
                base += f"  • {factor}\n"
            if bearish:
                if bullish:
                    base += "\n"
                base += "**Bearish Points:**\n"
                for factor in bearish[:2]:
                    base += f"  • {factor}\n"
            if not bullish and not bearish:
                base += "No strong technical signals present at this time.\n"
            base += "\n**Action:** Wait on the sidelines. Current conditions don't justify new positions. Monitor for clearer signals."

        return base

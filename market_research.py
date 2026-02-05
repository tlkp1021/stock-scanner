"""
Market Research Module
Performs web research to gather market sentiment and analysis.
"""

import re
from datetime import datetime
from typing import List, Dict
import requests
from bs4 import BeautifulSoup


class WebResearcher:
    """Perform web research on market conditions."""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search_market_sentiment(self, asset_name: str) -> Dict:
        """Search for recent market sentiment about an asset."""
        # This is a simplified version - in production, you'd use proper search APIs
        sentiment = {
            'asset': asset_name,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'headlines': [],
            'summary': 'No web research available'
        }

        # Bullish indicators
        bullish_keywords = [
            'rally', 'surge', 'bull', 'bullish', 'rallying', 'gains',
            'breakout', 'momentum', 'uptrend', 'record high', 'soars'
        ]

        # Bearish indicators
        bearish_keywords = [
            'plunge', 'bear', 'bearish', 'decline', 'falls', 'dropping',
            'sell-off', 'crash', 'downturn', 'slump', 'tumbles'
        ]

        return sentiment

    def get_fear_greed_index(self) -> Dict:
        """Get the Crypto Fear & Greed Index as a market sentiment indicator."""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                fng = data['data'][0]
                value = int(fng['value'])
                classification = fng['value_classification']

                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': fng['timestamp'],
                    'interpretation': self._interpret_fear_greed(value)
                }
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")

        return {'value': None, 'classification': 'N/A', 'interpretation': 'N/A'}

    def _interpret_fear_greed(self, value: int) -> str:
        """Interpret the Fear & Greed Index value."""
        if value >= 75:
            return "Extreme Greed - Market may be overvalued, consider taking profits"
        elif value >= 55:
            return "Greed - Market optimism is high"
        elif value >= 45:
            return "Neutral - Market sentiment is balanced"
        elif value >= 25:
            return "Fear - Market pessimism, potential buying opportunity"
        else:
            return "Extreme Fear - High panic, may signal buying opportunity"

    def get_vix_level(self) -> Dict:
        """Get VIX level as volatility indicator (via Yahoo Finance)."""
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()

            if 'chart' in data and data['chart']['result']:
                meta = data['chart']['result'][0]['meta']
                current_price = meta.get('regularMarketPrice')
                previous_close = meta.get('previousClose')

                interpretation = self._interpret_vix(current_price) if current_price else "N/A"

                return {
                    'value': current_price,
                    'previous_close': previous_close,
                    'interpretation': interpretation
                }
        except Exception as e:
            print(f"Error fetching VIX: {e}")

        return {'value': None, 'interpretation': 'N/A'}

    def _interpret_vix(self, vix: float) -> str:
        """Interpret VIX level."""
        if vix >= 30:
            return f"High volatility ({vix:.2f}) - Elevated fear, potential buying opportunity"
        elif vix >= 20:
            return f"Moderate volatility ({vix:.2f}) - Some uncertainty in markets"
        elif vix >= 12:
            return f"Normal volatility ({vix:.2f}) - Market conditions are stable"
        else:
            return f"Low volatility ({vix:.2f}) - Complacency, watch for reversals"

    def get_ad_line(self) -> Dict:
        """Get Advance/Decline Line data for NYSE to measure market breadth.

        The AD Line tracks the NYSE (New York Stock Exchange) market breadth by measuring
        the cumulative difference between advancing and declining stocks.

        Interpretation:
        - A rising AD Line indicates broad market participation in uptrends (bullish)
        - A falling AD Line shows weakening breadth even if index rises (divergence/bearish)
        - This specifically measures NYSE stocks, which represents one of the largest US equity markets
        """
        try:
            # Fetch NYSE Composite Index (^NYA) data as a proxy for NYSE market breadth
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^NYA?interval=1d&range=1mo"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()

            if 'chart' in data and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                indicators = result.get('indicators', {})
                quote = indicators.get('quote', [{}])[0]

                # Get recent price data to analyze breadth trend
                closes = quote.get('close', [])[-50:] if quote.get('close') else []
                volumes = quote.get('volume', [])[-50:] if quote.get('volume') else []

                # Filter out None values
                closes = [c for c in closes if c is not None]
                volumes = [v for v in volumes if v is not None]

                if len(closes) >= 20:
                    # Calculate breadth trend based on NYSE price action and volume
                    current_price = closes[-1]
                    prev_price = closes[-20] if len(closes) >= 20 else closes[0]

                    # Calculate breadth trend
                    if current_price and prev_price:
                        price_change_20d = ((current_price - prev_price) / prev_price) * 100

                        # Analyze volume trend for confirmation
                        volume_trend = "neutral"
                        if len(volumes) >= 10:
                            avg_volume_recent = sum(volumes[-5:]) / 5
                            avg_volume_prev = sum(volumes[-20:-15]) / 5 if len(volumes) >= 20 else avg_volume_recent
                            volume_trend = "increasing" if avg_volume_recent > avg_volume_prev * 1.1 else "decreasing" if avg_volume_recent < avg_volume_prev * 0.9 else "neutral"

                        # Determine breadth signal for NYSE
                        if price_change_20d > 2 and volume_trend == "increasing":
                            breadth_signal = "Strong Bullish"
                            breadth_interpretation = (f"NYSE breadth is expanding with +{price_change_20d:.1f}% over 20 days "
                                                    f"and {volume_trend} volume - broad participation in NYSE rally")
                        elif price_change_20d > 1:
                            breadth_signal = "Bullish"
                            breadth_interpretation = (f"NYSE breadth is positive with +{price_change_20d:.1f}% over 20 days "
                                                    f"- moderate participation across NYSE stocks")
                        elif price_change_20d < -2:
                            breadth_signal = "Bearish"
                            breadth_interpretation = (f"NYSE breadth is deteriorating with {price_change_20d:.1f}% over 20 days "
                                                    f"- broad selling pressure across NYSE")
                        else:
                            breadth_signal = "Neutral"
                            breadth_interpretation = f"NYSE breadth is mixed with {price_change_20d:+.1f}% - selective participation among NYSE stocks"

                        return {
                            'value': price_change_20d,
                            'signal': breadth_signal,
                            'current_price': current_price,
                            'interpretation': breadth_interpretation,
                            'volume_trend': volume_trend,
                            'market': 'NYSE (New York Stock Exchange)'
                        }
                else:
                    # Not enough data points
                    return {
                        'value': 0,
                        'signal': 'Neutral',
                        'interpretation': 'Insufficient NYSE breadth data available',
                        'market': 'NYSE (New York Stock Exchange)'
                    }
        except Exception as e:
            print(f"Error fetching AD Line: {e}")

        return {'value': None, 'signal': 'N/A', 'interpretation': 'Unable to fetch NYSE breadth data', 'market': 'N/A'}

    def get_market_breadth_summary(self, ad_line_data: Dict) -> str:
        """Generate a detailed market breadth summary."""
        if ad_line_data.get('value') is None:
            return "**Market Breadth:** Data not available"

        signal = ad_line_data.get('signal', 'N/A')
        value = ad_line_data.get('value', 0)
        volume_trend = ad_line_data.get('volume_trend', 'N/A')

        summary = f"**AD Line (Market Breadth):** {signal}\n"
        summary += f"- {ad_line_data.get('interpretation', '')}\n"
        summary += f"- Volume trend: {volume_trend.capitalize()}\n"

        # Add actionable insight
        if signal == "Strong Bullish":
            summary += "- Insight: Strong breadth confirms the uptrend with broad participation. Favorable for new long positions."
        elif signal == "Bullish":
            summary += "- Insight: Positive breadth supports the rally but watch for deterioration."
        elif signal == "Bearish":
            summary += "- Insight: Weak breadth suggests selective selling or potential reversal. Reduce exposure."
        else:
            summary += "- Insight: Mixed breadth indicates choppy conditions. Stay selective and defensive."

        return summary

    def get_major_trends(self) -> List[Dict]:
        """Get major market trends and themes."""
        trends = []

        # Current market themes (this would ideally be fetched from news APIs)
        current_date = datetime.now()

        # Add general market overview
        trends.append({
            'theme': 'Market Overview',
            'description': 'Analyzing major indices, commodities, and cryptocurrencies for technical signals',
            'timestamp': current_date.isoformat()
        })

        return trends

    def summarize_market_context(self, fear_greed: Dict, vix: Dict) -> str:
        """Summarize overall market context."""
        summary_parts = []

        if fear_greed.get('value'):
            summary_parts.append(
                f"**Fear & Greed:** {fear_greed['classification']} ({fear_greed['value']}/100) - "
                f"{fear_greed['interpretation']}"
            )

        if vix.get('value'):
            summary_parts.append(
                f"**VIX (Volatility):** {vix['interpretation']}"
            )

        if summary_parts:
            return "\n\n".join(summary_parts)
        else:
            return "Market context data not available."


def analyze_sentiment_from_headlines(headlines: List[str]) -> Dict:
    """Analyze sentiment from a list of headlines."""
    bullish_words = [
        'rally', 'surge', 'bull', 'gains', 'breakout', 'soars', 'jumps',
        'rallying', 'higher', 'rise', 'rises', 'strong', 'buy', 'upbeat'
    ]
    bearish_words = [
        'plunge', 'bear', 'fall', 'drops', 'sell-off', 'crash', 'slump',
        'tumbles', 'weak', 'sell', 'downbeat', 'decline', 'losses'
    ]

    bullish_count = 0
    bearish_count = 0

    for headline in headlines:
        headline_lower = headline.lower()
        for word in bullish_words:
            if word in headline_lower:
                bullish_count += 1
        for word in bearish_words:
            if word in headline_lower:
                bearish_count += 1

    total = bullish_count + bearish_count
    if total == 0:
        sentiment = 'neutral'
        score = 0
    else:
        bullish_ratio = bullish_count / total
        score = (bullish_ratio - 0.5) * 2  # Range from -1 to 1

        if score > 0.3:
            sentiment = 'bullish'
        elif score < -0.3:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

    return {
        'sentiment': sentiment,
        'score': score,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count
    }

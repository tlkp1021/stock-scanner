"""
Market Data Fetcher Module
Uses OpenBB to fetch market data for various assets.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time

try:
    from config import ASSETS
    CONFIG_ASSETS = ASSETS
except ImportError:
    CONFIG_ASSETS = None


class MarketDataFetcher:
    """Fetch market data using OpenBB."""

    def __init__(self, custom_assets: Dict = None):
        try:
            from openbb import obb
            self.obb = obb
            self.available = True
        except ImportError:
            print("Warning: OpenBB not installed. Run: pip install openbb")
            self.available = False
            self.obb = None

        # Use custom assets or config file
        self.assets_config = custom_assets or CONFIG_ASSETS or self._default_assets()

    def _default_assets(self) -> Dict:
        """Default assets to analyze if no config provided."""
        return {
            '^NYA': {'name': 'NYSE Composite Index', 'type': 'index', 'symbol': '^NYA', 'enabled': True},
            '^HSI': {'name': 'Hang Seng Index', 'type': 'index', 'symbol': '^HSI', 'enabled': True},
            'TSLA': {'name': 'Tesla Inc', 'type': 'stock', 'symbol': 'TSLA', 'enabled': True},
            'MSFT': {'name': 'Microsoft Corp', 'type': 'stock', 'symbol': 'MSFT', 'enabled': True},
            'BTC-USD': {'name': 'Bitcoin', 'type': 'crypto', 'symbol': 'BTC-USD', 'enabled': True},
            'GC=F': {'name': 'Gold Futures', 'type': 'commodity', 'symbol': 'GC=F', 'enabled': True},
            'TLT': {'name': 'iShares 20+ Year Treasury Bond ETF', 'type': 'etf', 'symbol': 'TLT', 'enabled': True},
        }

    def fetch_equity_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch equity/stock data."""
        if not self.available:
            return None

        try:
            data = self.obb.equity.price.historical(
                symbol=symbol,
                period=period,
                provider="yfinance"
            )
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            else:
                df = pd.DataFrame(data)

            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching equity data for {symbol}: {e}")
            return None

    def fetch_crypto_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch crypto data."""
        if not self.available:
            return None

        try:
            # OpenBB crypto uses different format
            data = self.obb.crypto.price.historical(
                symbol=symbol,
                period=period,
                provider="yfinance"  # Can also use other providers
            )
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            else:
                df = pd.DataFrame(data)

            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            # Fallback to yfinance directly
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                return df
            except:
                return None

    def fetch_index_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch index data."""
        if not self.available:
            return None

        try:
            # Use yfinance for indices as well
            data = self.obb.equity.price.historical(
                symbol=symbol,
                period=period,
                provider="yfinance"
            )
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            else:
                df = pd.DataFrame(data)

            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching index data for {symbol}: {e}")
            return None

    def fetch_commodity_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch commodity data (Gold, etc)."""
        if not self.available:
            return None

        try:
            data = self.obb.equity.price.historical(
                symbol=symbol,
                period=period,
                provider="yfinance"
            )
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            else:
                df = pd.DataFrame(data)

            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching commodity data for {symbol}: {e}")
            return None

    def fetch_all_assets(self) -> Dict[str, Dict]:
        """Fetch data for all configured assets."""
        assets = self.assets_config
        results = {}

        for key, asset in assets.items():
            # Skip if disabled
            if not asset.get('enabled', True):
                continue

            print(f"Fetching data for {asset['name']} ({key})...")
            asset_type = asset['type']

            if asset_type == 'crypto':
                df = self.fetch_crypto_data(key)
            elif asset_type == 'index':
                df = self.fetch_index_data(key)
            elif asset_type == 'commodity':
                df = self.fetch_commodity_data(key)
            else:  # stock or etf
                df = self.fetch_equity_data(key)

            if df is not None and not df.empty:
                asset['data'] = df
                asset['fetched_at'] = datetime.now().isoformat()
                results[key] = asset
                print(f"  ✓ Fetched {len(df)} data points")
            else:
                print(f"  ✗ Failed to fetch data")

            time.sleep(0.5)  # Rate limiting

        return results


class MarketNewsFetcher:
    """Fetch market news and analysis."""

    def __init__(self):
        try:
            from openbb import obb
            self.obb = obb
            self.available = True
        except ImportError:
            self.available = False
            self.obb = None

    def fetch_market_news(self, limit: int = 10) -> List[Dict]:
        """Fetch general market news."""
        news_items = []

        if not self.available:
            return news_items

        try:
            # Try different news sources
            news = self.obb.news.world(
                limit=limit
            )
            if hasattr(news, 'to_dataframe'):
                df = news.to_dataframe()
            else:
                df = pd.DataFrame(news)

            if not df.empty:
                for _, row in df.head(limit).iterrows():
                    news_items.append({
                        'title': row.get('title', 'N/A'),
                        'url': row.get('url', ''),
                        'published': row.get('published', 'N/A'),
                        'source': row.get('source', 'OpenBB')
                    })
        except Exception as e:
            print(f"Error fetching market news: {e}")

        return news_items

    def fetch_asset_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch news for specific asset."""
        news_items = []

        if not self.available:
            return news_items

        try:
            news = self.obb.news.company(
                symbol=symbol,
                limit=limit
            )
            if hasattr(news, 'to_dataframe'):
                df = news.to_dataframe()
            else:
                df = pd.DataFrame(news)

            if not df.empty:
                for _, row in df.head(limit).iterrows():
                    news_items.append({
                        'title': row.get('title', 'N/A'),
                        'url': row.get('url', ''),
                        'published': row.get('published', 'N/A'),
                        'source': row.get('source', 'OpenBB')
                    })
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")

        return news_items

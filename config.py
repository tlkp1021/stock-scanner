"""
Configuration for Daily Market Scanner
Edit this file to customize the assets and parameters.
"""

# Assets to analyze
ASSETS = {
    # ========== US Indices ==========
    '^GSPC': {  # S&P 500
        'name': 'S&P 500 Index',
        'type': 'index',
        'enabled': True
    },
    '^IXIC': {  # NASDAQ Composite
        'name': 'NASDAQ Composite Index',
        'type': 'index',
        'enabled': True
    },
    '^NDX': {  # NASDAQ 100
        'name': 'NASDAQ 100 Index',
        'type': 'index',
        'enabled': True
    },
    '^DJI': {  # Dow Jones Industrial Average
        'name': 'Dow Jones Industrial Average',
        'type': 'index',
        'enabled': True
    },
    '^NYA': {  # NYSE Composite
        'name': 'NYSE Composite Index',
        'type': 'index',
        'enabled': True
    },
    '^RUT': {  # Russell 2000 (Small Cap)
        'name': 'Russell 2000 Index',
        'type': 'index',
        'enabled': False
    },
    '^VIX': {  # VIX (Volatility Index)
        'name': 'VIX Volatility Index',
        'type': 'index',
        'enabled': False
    },

    # ========== Global Indices ==========
    '^HSI': {  # Hong Kong
        'name': 'Hang Seng Index',
        'type': 'index',
        'enabled': True
    },
    '^N225': {  # Japan
        'name': 'Nikkei 225',
        'type': 'index',
        'enabled': True
    },
    '^FTSE': {  # UK FTSE 100
        'name': 'FTSE 100 Index',
        'type': 'index',
        'enabled': False
    },
    '^GDAXI': {  # Germany DAX
        'name': 'DAX Performance Index',
        'type': 'index',
        'enabled': False
    },
    '^FCHI': {  # France CAC 40
        'name': 'CAC 40 Index',
        'type': 'index',
        'enabled': False
    },

    # ========== Stocks (Tech Giants) ==========
    'NVDA': {
        'name': 'NVIDIA Corp',
        'type': 'stock',
        'enabled': True
    },
    'AAPL': {
        'name': 'Apple Inc',
        'type': 'stock',
        'enabled': True
    },
    'MSFT': {
        'name': 'Microsoft Corp',
        'type': 'stock',
        'enabled': True
    },
    'GOOGL': {
        'name': 'Alphabet Inc (Google)',
        'type': 'stock',
        'enabled': True
    },
    'AMZN': {
        'name': 'Amazon.com Inc',
        'type': 'stock',
        'enabled': True
    },
    'META': {
        'name': 'Meta Platforms Inc',
        'type': 'stock',
        'enabled': True
    },
    'TSLA': {
        'name': 'Tesla Inc',
        'type': 'stock',
        'enabled': True
    },

    # ========== Crypto ==========
    'BTC-USD': {
        'name': 'Bitcoin',
        'type': 'crypto',
        'enabled': True
    },
    'ETH-USD': {
        'name': 'Ethereum',
        'type': 'crypto',
        'enabled': True
    },

    # ========== Commodities ==========
    'GC=F': {
        'name': 'Gold Futures',
        'type': 'commodity',
        'enabled': True
    },
    'SI=F': {
        'name': 'Silver Futures',
        'type': 'commodity',
        'enabled': True
    },
    'CL=F': {  # Crude Oil
        'name': 'Crude Oil Futures',
        'type': 'commodity',
        'enabled': True
    },

    # ========== ETFs ==========
    'TLT': {  # Long-term Treasury
        'name': 'iShares 20+ Year Treasury Bond ETF',
        'type': 'etf',
        'enabled': True
    },
    'SPY': {  # S&P 500 ETF
        'name': 'SPDR S&P 500 ETF',
        'type': 'etf',
        'enabled': True
    },
    'QQQ': {  # NASDAQ 100 ETF
        'name': 'Invesco QQQ Trust (NASDAQ 100 ETF)',
        'type': 'etf',
        'enabled': True
    },
    'IWM': {  # Russell 2000 ETF
        'name': 'iShares Russell 2000 ETF',
        'type': 'etf',
        'enabled': True
    },
}

# Technical Analysis Parameters
TA_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'sma_short': 20,
    'sma_medium': 50,
    'sma_long': 200,
    'bb_period': 20,
    'bb_std': 2,
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'atr_period': 14,
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    'strong_buy': 5,
    'buy': 3,
    'weak_buy': 1,
    'weak_sell': -1,
    'sell': -3,
    'strong_sell': -5,
}

# Report Settings
REPORT_SETTINGS = {
    'default_output_dir': './reports',
    'include_news': True,
    'news_limit': 15,
    'max_signal_reasons': 5,
}

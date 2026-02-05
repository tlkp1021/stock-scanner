# Daily Market Scanner

A Python program that performs daily technical analysis on stocks, indices, crypto, and commodities to generate buy/sell signals.

## Features

- **Multi-Asset Analysis**: Analyze stocks, indices, cryptocurrencies, and commodities
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Moving Averages (SMA, EMA)
  - Bollinger Bands
  - Stochastic Oscillator
  - Williams %R
  - CCI (Commodity Channel Index)
  - ATR (Average True Range)
- **Market Context**: Fear & Greed Index, VIX volatility
- **News Integration**: Fetches latest market news
- **HTML Reports**: Beautiful, interactive HTML reports

## Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install OpenBB Platform (if not already installed):**

```bash
pip install openbb
```

## Usage

### Basic Usage

Run the scanner with default settings:

```bash
python main.py
```

### Command Line Options

```bash
# Specify output file
python main.py -o my_report.html

# Verbose output
python main.py -v

# Skip news fetching (faster)
python main.py --no-news
```

### Output

The program generates an HTML report file that you can open in any web browser. The report includes:

- Market context (Fear & Greed Index, VIX)
- Individual asset analysis with technical indicators
- Buy/Sell/Hold signals with confidence levels
- Detailed reasoning for each signal
- Recent market news

### Scheduling Daily Runs

#### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create a new task
3. Set trigger to daily at your preferred time
4. Set action to run: `python C:\path\to\stock_scanner\main.py`

#### Linux/Mac (Cron)

```bash
# Edit crontab
crontab -e

# Add daily run at 8:00 AM
0 8 * * * cd /path/to/stock_scanner && python main.py
```

## Configuration

Edit `config.py` to customize:

- **Assets**: Add/remove assets to analyze
- **Technical Parameters**: Adjust indicator periods
- **Signal Thresholds**: Modify buy/sell sensitivity

## Example Report

The HTML report includes:
- Price and daily change
- Technical indicator values
- Overall trading signal (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- Detailed signal breakdown
- Color-coded visual indicators

## Disclaimer

This software is for informational purposes only and does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## License

MIT License

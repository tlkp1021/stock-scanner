#!/usr/bin/env python3
"""
Daily Market Scanner
Main program that performs technical analysis and generates trading signals.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from market_data import MarketDataFetcher, MarketNewsFetcher
from market_research import WebResearcher
from technical_analysis import SignalAnalyzer
from report_generator import ReportGenerator


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Daily Market Scanner - Technical Analysis & Trading Signals')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HTML file path (default: market_report_TIMESTAMP.html)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-news', action='store_true',
                        help='Skip fetching news (faster)')

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Daily Market Scanner Starting...")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize components
    print("Initializing components...")
    data_fetcher = MarketDataFetcher()
    news_fetcher = MarketNewsFetcher()
    researcher = WebResearcher()
    signal_analyzer = SignalAnalyzer()
    report_gen = ReportGenerator()

    if not data_fetcher.available:
        print("❌ Error: OpenBB is not properly installed.")
        print("Please run: pip install openbb")
        sys.exit(1)

    # Step 1: Fetch market data
    print("\n" + "=" * 60)
    print("📡 Step 1: Fetching Market Data")
    print("=" * 60)

    assets_data = data_fetcher.fetch_all_assets()

    if not assets_data:
        print("❌ Error: No market data was fetched. Please check your internet connection.")
        sys.exit(1)

    print(f"\n✅ Successfully fetched data for {len(assets_data)} assets")

    # Step 2: Perform technical analysis
    print("\n" + "=" * 60)
    print("📊 Step 2: Performing Technical Analysis")
    print("=" * 60)

    analysis_results = {}

    for symbol, asset in assets_data.items():
        print(f"\nAnalyzing {asset['name']} ({symbol})...")

        df = asset['data']
        analysis = signal_analyzer.analyze(df)

        if 'error' not in analysis:
            signal = signal_analyzer.generate_signals(analysis, df)

            print(f"  Price: ${analysis['price']:.2f}")
            print(f"  Change: {analysis['price_change_pct']:+.2f}%")
            print(f"  RSI: {analysis['rsi']:.1f}")
            print(f"  Signal: {signal['signal']} (score: {signal['confidence']:+.2f})")

            # Prepare chart data (last 365 days)
            chart_df = df.tail(365).copy()

            # Convert dates to ISO format for JSON serialization
            chart_dates = []
            for idx in chart_df.index:
                if hasattr(idx, 'strftime'):
                    chart_dates.append(idx.strftime('%Y-%m-%d'))
                else:
                    chart_dates.append(str(idx))

            chart_closes = chart_df['close'].tolist()
            chart_volumes = chart_df['volume'].tolist() if 'volume' in chart_df.columns else []

            analysis_results[symbol] = {
                'data': df,
                'asset_info': {
                    'name': asset['name'],
                    'type': asset['type']
                },
                'analysis': analysis,
                'signal': signal,
                'last_update': df.index[-1] if hasattr(df.index[-1], 'strftime') else datetime.now(),
                'chart_data': {
                    'dates': chart_dates,
                    'closes': chart_closes,
                    'volumes': chart_volumes,
                    'period_days': 365
                }
            }
        else:
            print(f"  ⚠️ {analysis['error']}")

    # Step 3: Market research
    print("\n" + "=" * 60)
    print("🔍 Step 3: Market Research")
    print("=" * 60)

    market_context = {}

    # Fear & Greed Index
    print("\nFetching Fear & Greed Index...")
    fear_greed = researcher.get_fear_greed_index()
    if fear_greed.get('value'):
        print(f"  📊 Fear & Greed: {fear_greed['classification']} ({fear_greed['value']}/100)")
    market_context['fear_greed'] = fear_greed

    # VIX
    print("\nFetching VIX (Volatility Index)...")
    vix = researcher.get_vix_level()
    if vix.get('value'):
        print(f"  📈 VIX: {vix['value']:.2f}")
    market_context['vix'] = vix

    # AD Line (Market Breadth)
    print("\nFetching AD Line (Market Breadth)...")
    ad_line = researcher.get_ad_line()
    if ad_line.get('value') is not None:
        print(f"  📊 AD Line: {ad_line['signal']} ({ad_line['value']:+.1f}%)")
    market_context['ad_line'] = ad_line

    # News
    news_items = []
    if not args.no_news:
        print("\nFetching market news...")
        news_items = news_fetcher.fetch_market_news(limit=15)
        if news_items:
            print(f"  📰 Fetched {len(news_items)} news items")

    # Step 4: Generate report
    print("\n" + "=" * 60)
    print("📝 Step 4: Generating Report")
    print("=" * 60)

    html = report_gen.generate(
        analysis_results=analysis_results,
        market_context=market_context,
        news_items=news_items
    )

    # Save report
    output_file = args.output
    if output_file is None:
        output_file = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    saved_path = report_gen.save(html, output_file)

    print(f"\n✅ Report saved to: {saved_path}")
    print(f"📂 Full path: {Path(saved_path).absolute()}")

    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)

    buy_signals = sum(1 for s in analysis_results.values()
                      if 'buy' in s.get('signal', {}).get('signal', '').lower())
    sell_signals = sum(1 for s in analysis_results.values()
                       if 'sell' in s.get('signal', {}).get('signal', '').lower())
    hold_signals = len(analysis_results) - buy_signals - sell_signals

    print(f"\nTotal Assets Analyzed: {len(analysis_results)}")
    print(f"  Buy Signals: {buy_signals}")
    print(f"  Sell Signals: {sell_signals}")
    print(f"  Hold/Neutral: {hold_signals}")

    if fear_greed.get('value'):
        print(f"\nMarket Sentiment: {fear_greed['classification']}")

    print("\n" + "=" * 60)
    print("✅ Daily Market Scanner Complete!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

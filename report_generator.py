"""
HTML Report Generator Module
Creates an HTML report summarizing all analysis.
"""

from datetime import datetime
from typing import Dict, List
from jinja2 import Template


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Market Scanner Report - {{ report_date }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e0e0e0;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header .subtitle {
            color: #888;
            font-size: 1.1em;
        }

        .market-context {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .market-context h2 {
            color: #00d9ff;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .context-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .context-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00d9ff;
        }

        .context-item .label {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .context-item .value {
            font-size: 1.2em;
            font-weight: 600;
        }

        .signals-summary {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .signals-summary h2 {
            color: #00ff88;
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        .signal-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }

        .asset-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .asset-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .asset-card .asset-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .asset-card .asset-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #fff;
        }

        .asset-card .asset-symbol {
            color: #888;
            font-size: 0.9em;
        }

        .asset-card .asset-type {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            text-transform: uppercase;
        }

        .asset-card .type-stock { background: rgba(59, 130, 246, 0.3); color: #60a5fa; }
        .asset-card .type-crypto { background: rgba(251, 191, 36, 0.3); color: #fbbf24; }
        .asset-card .type-index { background: rgba(139, 92, 246, 0.3); color: #a78bfa; }
        .asset-card .type-commodity { background: rgba(234, 179, 8, 0.3); color: #eab308; }
        .asset-card .type-etf { background: rgba(34, 197, 94, 0.3); color: #4ade80; }

        .asset-card .price-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .asset-card .price {
            font-size: 1.8em;
            font-weight: 700;
            color: #fff;
        }

        .asset-card .price-change {
            font-size: 1em;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 8px;
        }

        .price-change.positive { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .price-change.negative { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .price-change.neutral { background: rgba(255, 255, 255, 0.1); color: #888; }

        .asset-card .signal-badge {
            text-align: center;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-weight: 700;
            font-size: 1.1em;
        }

        .signal-badge.strong-buy { background: linear-gradient(135deg, #059669, #10b981); color: #fff; }
        .signal-badge.buy { background: linear-gradient(135deg, #10b981, #34d399); color: #fff; }
        .signal-badge.weak-buy { background: rgba(16, 185, 129, 0.3); color: #6ee7b7; }
        .signal-badge.hold { background: rgba(255, 255, 255, 0.1); color: #888; }
        .signal-badge.weak-sell { background: rgba(239, 68, 68, 0.3); color: #fca5a5; }
        .signal-badge.sell { background: linear-gradient(135deg, #ef4444, #f87171); color: #fff; }
        .signal-badge.strong-sell { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; }

        .asset-card .indicators {
            margin-top: 15px;
        }

        .asset-card .indicator-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9em;
        }

        .asset-card .indicator-label {
            color: #888;
        }

        .asset-card .indicator-value {
            color: #e0e0e0;
        }

        .asset-card .signal-reasons {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .asset-card .signal-reasons h4 {
            color: #00d9ff;
            margin-bottom: 10px;
            font-size: 0.9em;
        }

        .asset-card .price-datetime {
            color: #666;
            font-size: 0.75em;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 15px;
        }

        .asset-card .signal-explanation {
            margin-top: 15px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-size: 0.8em;
            color: #bbb;
            line-height: 1.4;
        }

        .asset-card .signal-explanation strong {
            color: #fff;
        }

        .asset-card .signal-explanation ul {
            margin: 8px 0;
            padding-left: 20px;
        }

        .asset-card .signal-explanation li {
            margin: 4px 0;
        }

        .asset-card .signal-reasons ul {
            list-style: none;
            padding: 0;
        }

        .asset-card .signal-reasons li {
            padding: 5px 0;
            font-size: 0.85em;
            color: #aaa;
            padding-left: 15px;
            position: relative;
        }

        .asset-card .signal-reasons li:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #00d9ff;
        }

        .chart-patterns {
            margin-top: 15px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border-left: 3px solid #fbbf24;
        }

        .chart-patterns h5 {
            color: #fbbf24;
            margin: 0 0 8px 0;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pattern-item {
            display: flex;
            align-items: center;
            padding: 4px 0;
            font-size: 0.75em;
        }

        .pattern-name {
            font-weight: 600;
            margin-right: 8px;
        }

        .pattern-bullish {
            color: #4ade80;
        }

        .pattern-bearish {
            color: #f87171;
        }

        .pattern-neutral {
            color: #888;
        }

        .pattern-reliability {
            margin-left: auto;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7em;
        }

        .pattern-reliability.high {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }

        .pattern-reliability.medium {
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
        }

        .pattern-reliability.low {
            background: rgba(107, 114, 128, 0.2);
            color: #888;
        }

        .no-patterns {
            color: #666;
            font-size: 0.75em;
            font-style: italic;
        }

        .news-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .news-section h2 {
            color: #fbbf24;
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        .news-item {
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #fbbf24;
        }

        .news-item .title {
            color: #fff;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .news-item .meta {
            color: #888;
            font-size: 0.85em;
        }

        .news-item a {
            color: #00d9ff;
            text-decoration: none;
        }

        .news-item a:hover {
            text-decoration: underline;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }

        .disclaimer {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.85em;
            color: #fca5a5;
        }

        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .stat-box {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .stat-box .value {
            font-size: 1.8em;
            font-weight: 700;
            color: #00ff88;
        }

        .stat-box .label {
            color: #888;
            font-size: 0.85em;
            margin-top: 5px;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }

            .signal-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Chart modal styles */
        .chart-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .chart-modal.active {
            display: flex;
        }

        .chart-modal-content {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 30px;
            max-width: 900px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }

        .chart-modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-modal-title {
            font-size: 1.5em;
            font-weight: 600;
            color: #00d9ff;
        }

        .chart-modal-close {
            background: rgba(239, 68, 68, 0.3);
            color: #f87171;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 1.2em;
            cursor: pointer;
            transition: background 0.2s;
        }

        .chart-modal-close:hover {
            background: rgba(239, 68, 68, 0.5);
        }

        .chart-container {
            height: 400px;
            position: relative;
        }

        .asset-card {
            cursor: pointer;
        }

        .asset-card:hover {
            border-color: rgba(0, 217, 255, 0.3);
        }

        .chart-hint {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 217, 255, 0.2);
            color: #00d9ff;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.7em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Market Scanner Report</h1>
            <p class="subtitle">Generated on {{ report_date }} at {{ report_time }}</p>
        </div>

        <div class="market-context">
            <h2>🌐 Market Context</h2>
            <div class="context-grid">
                {% if fear_greed.value is not none %}
                <div class="context-item">
                    <div class="label">Fear & Greed Index</div>
                    <div class="value">{{ fear_greed.classification }} ({{ fear_greed.value }}/100)</div>
                    <div style="margin-top: 5px; font-size: 0.85em; color: #888;">{{ fear_greed.interpretation }}</div>
                </div>
                {% endif %}
                {% if vix.value is not none %}
                <div class="context-item">
                    <div class="label">VIX (Volatility Index)</div>
                    <div class="value">{{ "%.2f"|format(vix.value) }}</div>
                    <div style="margin-top: 5px; font-size: 0.85em; color: #888;">{{ vix.interpretation }}</div>
                </div>
                {% endif %}
                {% if ad_line.value is not none %}
                <div class="context-item">
                    <div class="label">AD Line (Market Breadth)</div>
                    <div class="value">{{ ad_line.signal }}</div>
                    <div style="margin-top: 5px; font-size: 0.85em; color: #888;">{{ ad_line.interpretation }}</div>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="signals-summary">
            <h2>📈 Trading Signals</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="value" style="color: #10b981;">{{ buy_count }}</div>
                    <div class="label">Buy Signals</div>
                </div>
                <div class="stat-box">
                    <div class="value" style="color: #888;">{{ hold_count }}</div>
                    <div class="label">Hold/Neutral</div>
                </div>
                <div class="stat-box">
                    <div class="value" style="color: #ef4444;">{{ sell_count }}</div>
                    <div class="label">Sell Signals</div>
                </div>
            </div>
        </div>

        <div class="signals-summary">
            <h2>🎯 Detailed Asset Analysis</h2>
            <div class="signal-grid">
                {% for asset in assets %}
                <div class="asset-card" onclick="openChartModal('{{ asset.symbol }}')">
                    <div class="asset-header">
                        <div>
                            <div class="asset-name">{{ asset.name }}</div>
                            <div class="asset-symbol">{{ asset.symbol }}</div>
                        </div>
                        <span class="asset-type type-{{ asset.type }}">{{ asset.type }}</span>
                        <div class="chart-hint">Click for 365-day chart</div>
                    </div>

                    <div class="price-info">
                        <div class="price">{{ asset.price_formatted }}</div>
                        <div class="price-change {{ asset.change_class }}">
                            {{ asset.change_formatted }}
                        </div>
                    </div>

                    <div class="price-datetime">
                        Price as of: {{ asset.price_datetime }}
                    </div>

                    <div class="signal-badge {{ asset.signal_class }}">
                        {{ asset.signal }}
                    </div>

                    <div class="indicators">
                        <div class="indicator-row">
                            <span class="indicator-label">RSI (14)</span>
                            <span class="indicator-value">{{ "%.1f"|format(asset.rsi) if asset.rsi else 'N/A' }}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-label">MACD</span>
                            <span class="indicator-value">{{ "%.2f"|format(asset.macd) if asset.macd else 'N/A' }}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-label">Price vs MA20</span>
                            <span class="indicator-value">{{ asset.vs_ma20 }}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-label">Price vs MA200</span>
                            <span class="indicator-value">{{ asset.vs_ma200 }}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-label">Stochastic %K</span>
                            <span class="indicator-value">{{ "%.1f"|format(asset.stoch_k) if asset.stoch_k else 'N/A' }}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-label">Bollinger Position</span>
                            <span class="indicator-value">{{ "%.1f%%"|format(asset.bb_position) if asset.bb_position else 'N/A' }}</span>
                        </div>
                    </div>

                    {% if asset.explanation %}
                    <div class="signal-explanation">
                        {{ asset.explanation|safe }}
                    </div>
                    {% endif %}

                    {% if asset.reasons %}
                    <div class="signal-reasons">
                        <h4>Key Signals:</h4>
                        <ul>
                            {% for reason in asset.reasons[:5] %}
                            <li>{{ reason }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}

                    {% if asset.patterns %}
                    <div class="chart-patterns">
                        <h5>Chart Patterns Detected:</h5>
                        {% for pattern in asset.patterns[:4] %}
                        <div class="pattern-item">
                            <span class="pattern-name pattern-{{ pattern.sentiment }}">{{ pattern.name }}</span>
                            <span class="pattern-reliability {{ pattern.reliability }}">{{ pattern.reliability|capitalize }}</span>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>

        {% if news_items %}
        <div class="news-section">
            <h2>📰 Recent Market News</h2>
            {% for item in news_items[:10] %}
            <div class="news-item">
                <div class="title">{{ item.title }}</div>
                <div class="meta">
                    {% if item.published %}{{ item.published }} | {% endif %}
                    {% if item.url %}<a href="{{ item.url }}" target="_blank">Read more</a>{% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Report generated by Stock Scanner</p>
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute
                financial advice. Always do your own research and consult with a qualified financial advisor
                before making investment decisions. Past performance does not guarantee future results.
            </div>
        </div>

        <!-- Chart Modal -->
        <div class="chart-modal" id="chartModal">
            <div class="chart-modal-content">
                <div class="chart-modal-header">
                    <div class="chart-modal-title" id="chartTitle">Price Chart</div>
                    <button class="chart-modal-close" onclick="closeChartModal()">&times;</button>
                </div>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script>
            // Store all chart data
            const assetsData = {
                {% for asset in assets %}
                '{{ asset.symbol }}': {
                    name: '{{ asset.name }}',
                    symbol: '{{ asset.symbol }}',
                    dates: {{ asset.chart_dates|tojson if asset.chart_dates else [] }},
                    closes: {{ asset.chart_closes|tojson if asset.chart_closes else [] }},
                    volumes: {{ asset.chart_volumes|tojson if asset.chart_volumes else [] }}
                }{% if not loop.last %},{% endif %}
                {% endfor %}
            };

            let priceChart = null;

            function openChartModal(symbol) {
                const modal = document.getElementById('chartModal');
                const title = document.getElementById('chartTitle');
                const asset = assetsData[symbol];

                if (!asset) return;

                title.textContent = `${asset.name} (${asset.symbol}) - 365 Day Trend`;

                // Destroy existing chart
                if (priceChart) {
                    priceChart.destroy();
                }

                // Create gradient
                const ctx = document.getElementById('priceChart').getContext('2d');
                const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                gradient.addColorStop(0, 'rgba(0, 217, 255, 0.3)');
                gradient.addColorStop(1, 'rgba(0, 217, 255, 0)');

                // Create new chart
                priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: asset.dates,
                        datasets: [{
                            label: 'Price',
                            data: asset.closes,
                            borderColor: '#00d9ff',
                            backgroundColor: gradient,
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 3,
                            pointBackgroundColor: '#00d9ff',
                            pointBorderColor: '#fff',
                            pointHoverRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#00d9ff',
                                bodyColor: '#fff',
                                borderColor: '#00d9ff',
                                borderWidth: 1,
                                padding: 12,
                                displayColors: false,
                                callbacks: {
                                    label: function(context) {
                                        return 'Price: $' + parseFloat(context.raw).toFixed(2);
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                },
                                ticks: {
                                    color: '#888',
                                    maxRotation: 45,
                                    minRotation: 45
                                }
                            },
                            y: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                },
                                ticks: {
                                    color: '#888',
                                    callback: function(value) {
                                        return '$' + value.toFixed(0);
                                    }
                                }
                            }
                        }
                    }
                });

                modal.classList.add('active');
            }

            function closeChartModal() {
                const modal = document.getElementById('chartModal');
                modal.classList.remove('active');
            }

            // Close modal when clicking outside
            document.getElementById('chartModal').addEventListener('click', function(e) {
                if (e.target === this) {
                    closeChartModal();
                }
            });

            // Close modal with Escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    closeChartModal();
                }
            });
        </script>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generate HTML reports from analysis results."""

    def __init__(self):
        self.template = Template(HTML_TEMPLATE)

    def generate(self, analysis_results: Dict, market_context: Dict = None,
                 news_items: List = None) -> str:
        """Generate HTML report."""
        now = datetime.now()

        # Process assets for display
        assets_data = []
        buy_count = 0
        sell_count = 0
        hold_count = 0

        for symbol, data in analysis_results.items():
            if 'error' in data or 'analysis' not in data:
                continue

            analysis = data['analysis']
            signal_data = data.get('signal', {})
            asset_info = data.get('asset_info', {})

            # Format price
            price = analysis.get('price', 0)
            if price >= 1000:
                price_formatted = f"${price:,.2f}"
            elif price >= 1:
                price_formatted = f"${price:.2f}"
            else:
                price_formatted = f"${price:.4f}"

            # Format change
            change_pct = analysis.get('price_change_pct', 0)
            if change_pct >= 0:
                change_formatted = f"+{change_pct:.2f}%"
                change_class = "positive"
            else:
                change_formatted = f"{change_pct:.2f}%"
                change_class = "negative"

            # Signal class
            signal = signal_data.get('signal', 'HOLD/NEUTRAL')
            signal_lower = signal.lower().replace('/', '-').replace(' ', '-')
            if 'strong' in signal_lower:
                signal_class = f"{'strong-buy' if 'buy' in signal_lower else 'strong-sell'}"
            elif 'weak' in signal_lower:
                signal_class = f"{'weak-buy' if 'buy' in signal_lower else 'weak-sell'}"
            elif 'buy' in signal_lower:
                signal_class = 'buy'
            elif 'sell' in signal_lower:
                signal_class = 'sell'
            else:
                signal_class = 'hold'

            # Count signals
            if 'buy' in signal_lower:
                buy_count += 1
            elif 'sell' in signal_lower:
                sell_count += 1
            else:
                hold_count += 1

            # Calculate position relative to MAs
            current_price = analysis.get('price', 0)
            ma20 = analysis.get('sma_20', current_price)
            ma200 = analysis.get('sma_200', current_price)

            vs_ma20 = f"{((current_price / ma20 - 1) * 100):.1f}%" if ma20 > 0 else "N/A"
            vs_ma200 = f"{((current_price / ma200 - 1) * 100):.1f}%" if ma200 > 0 else "N/A"

            # Bollinger Band position
            bb_upper = analysis.get('bb_upper', current_price * 1.1)
            bb_lower = analysis.get('bb_lower', current_price * 0.9)
            if bb_upper > bb_lower:
                bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            else:
                bb_position = None

            # Format datetime
            last_update = data.get('last_update', datetime.now())
            if hasattr(last_update, 'strftime'):
                price_datetime = last_update.strftime("%Y-%m-%d %H:%M:%S")
            else:
                price_datetime = str(last_update)

            # Get explanation
            explanation = signal_data.get('explanation', '')
            # Convert markdown-style formatting to HTML
            if explanation:
                import re
                # Convert **text** to <strong>text</strong>
                explanation = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', explanation)
                # Convert newlines to <br>
                explanation = explanation.replace('\n', '<br>')

            # Get chart data
            chart_data = data.get('chart_data', {})

            assets_data.append({
                'name': asset_info.get('name', symbol),
                'symbol': symbol,
                'type': asset_info.get('type', 'stock'),
                'price_formatted': price_formatted,
                'change_formatted': change_formatted,
                'change_class': change_class,
                'signal': signal,
                'signal_class': signal_class,
                'rsi': analysis.get('rsi'),
                'macd': analysis.get('macd'),
                'stoch_k': analysis.get('stoch_k'),
                'bb_position': bb_position,
                'vs_ma20': vs_ma20,
                'vs_ma200': vs_ma200,
                'reasons': signal_data.get('reasons', []),
                'explanation': explanation,
                'price_datetime': price_datetime,
                'chart_dates': chart_data.get('dates', []),
                'chart_closes': chart_data.get('closes', []),
                'chart_volumes': chart_data.get('volumes', []),
                'patterns': analysis.get('patterns', [])
            })

        # Sort assets by signal strength
        signal_order = {'strong-buy': 0, 'buy': 1, 'weak-buy': 2,
                       'hold': 3, 'weak-sell': 4, 'sell': 5, 'strong-sell': 6}
        assets_data.sort(key=lambda x: signal_order.get(x['signal_class'], 3))

        # Prepare market context
        fear_greed = market_context.get('fear_greed', {}) if market_context else {}
        vix = market_context.get('vix', {}) if market_context else {}
        ad_line = market_context.get('ad_line', {}) if market_context else {}

        html = self.template.render(
            report_date=now.strftime("%Y-%m-%d"),
            report_time=now.strftime("%H:%M:%S"),
            fear_greed=fear_greed,
            vix=vix,
            ad_line=ad_line,
            assets=assets_data,
            news_items=news_items or [],
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count
        )

        return html

    def save(self, html: str, filename: str = None) -> str:
        """Save HTML report to file."""
        if filename is None:
            now = datetime.now()
            filename = f"market_report_{now.strftime('%Y%m%d_%H%M%S')}.html"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)

        return filename

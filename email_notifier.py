"""
Email Notifier Module
ส่งอีเมลแจ้งเตือนผลการวิเคราะห์ Trading Bot
"""

import re
import smtplib
import markdown
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailNotifier:
    """ส่งอีเมลแจ้งเตือนผลการวิเคราะห์"""

    def __init__(self, sender_email, sender_password, recipient_email):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def create_html_email(self, console_output, mode):
        """สร้าง HTML email จาก console output"""

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = self._parse_and_convert(console_output)

        # กำหนด icon ตาม mode
        mode_icon = "📅" if mode.lower() == "weekly" else "🌙"
        mode_text = "Weekly" if mode.lower() == "weekly" else "Monthly"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            padding: 2px;
            margin: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 700px;
            margin: 0 auto;
            background-color: #1e2a4a;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }}
        .header {{
            background: linear-gradient(135deg, #e94560 0%, #c73e54 100%);
            padding: 4px;
            text-align: center;
        }}
        .header h1 {{
            color: #fff;
            margin: 0;
            font-size: 22px;
            font-weight: 700;
        }}
        .header .subtitle {{
            color: rgba(255,255,255,0.85);
            font-size: 13px;
            margin-top: 8px;
        }}
        .header .mode-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 6px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 12px;
        }}
        .main-content {{ padding: 20px; }}
        .section {{
            background: #253557;
            border-radius: 12px;
            padding: 6px;
            margin-bottom: 16px;
        }}
        .section-title {{
            font-size: 15px;
            font-weight: 700;
            color: #4dabf7;
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #3a4a6b;
        }}
        .price-display {{
            background: linear-gradient(135deg, #2d3a5a 0%, #1e2a4a 100%);
            border-radius: 12px;
            padding: 4px;
            text-align: center;
            margin-bottom: 16px;
            border: 1px solid #3a4a6b;
        }}
        .price-label {{
            font-size: 12px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .price-value {{
            font-size: 32px;
            font-weight: 700;
            color: #ffd700;
            margin-top: 4px;
        }}
        .signal-box {{
            border-radius: 12px;
            padding: 4px;
            text-align: center;
            margin-bottom: 16px;
        }}
        .signal-long {{
            background: linear-gradient(135deg, #0d4a3a 0%, #1a6b50 100%);
            border: 2px solid #34e89e;
        }}
        .signal-short {{
            background: linear-gradient(135deg, #4a1a1a 0%, #6b2a2a 100%);
            border: 2px solid #e94560;
        }}
        .signal-wait {{
            background: linear-gradient(135deg, #4a4a1a 0%, #6b6b2a 100%);
            border: 2px solid #ffd700;
        }}
        .signal-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .signal-subtitle {{ font-size: 14px; opacity: 0.9; }}
        .indicator-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .indicator-item {{
            background: #1e2a4a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 4px;
            border-left: 3px solid #4dabf7;
        }}
        .indicator-label {{
            font-size: 10px;
            color: #8892b0;
            text-transform: uppercase;
        }}
        .indicator-value {{
            font-size: 14px;
            font-weight: 600;
            margin-top: 2px;
        }}
        .signal-list {{
            margin: 0;
            padding: 0;
            list-style: none;
        }}
        .signal-list li {{
            padding: 8px 10px;
            margin-bottom: 6px;
            border-radius: 6px;
            font-size: 12px;
        }}
        .signal-list .long {{
            background: rgba(52, 232, 158, 0.15);
            border-left: 3px solid #34e89e;
            color: #34e89e;
        }}
        .signal-list .short {{
            background: rgba(233, 69, 96, 0.15);
            border-left: 3px solid #e94560;
            color: #e94560;
        }}
        .signal-list .neutral {{
            background: rgba(255, 215, 0, 0.15);
            border-left: 3px solid #ffd700;
            color: #ffd700;
        }}
        .trade-setup {{
            background: #1e2a4a;
            border-radius: 8px;
            overflow: hidden;
        }}
        .trade-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 14px;
            border-bottom: 1px solid #3a4a6b;
        }}
        .trade-row:last-child {{ border-bottom: none; }}
        .trade-row.entry {{ background: #2a3a5a; }}
        .trade-row.sl {{ background: rgba(233, 69, 96, 0.1); }}
        .trade-row.tp {{ background: rgba(52, 232, 158, 0.1); }}
        .trade-label {{ font-size: 12px; color: #8892b0; }}
        .trade-value {{ font-size: 13px; font-weight: 600; }}
        .trade-value.green {{ color: #34e89e; }}
        .trade-value.red {{ color: #e94560; }}
        .trade-value.yellow {{ color: #ffd700; }}
        .progress-bar {{
            height: 8px;
            background: #1e2a4a;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            margin-top: 12px;
        }}
        .progress-long {{
            background: linear-gradient(90deg, #34e89e, #4ade80);
            height: 100%;
        }}
        .progress-short {{
            background: linear-gradient(90deg, #e94560, #f87171);
            height: 100%;
        }}
        .progress-labels {{
            display: flex;
            justify-content: space-between;
            margin-top: 6px;
            font-size: 11px;
        }}
        .level-list {{
            margin: 0;
            padding: 0;
            list-style: none;
        }}
        .level-list li {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #3a4a6b;
            font-size: 12px;
        }}
        .level-list li:last-child {{ border-bottom: none; }}
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .risk-low {{ background: #0d4a3a; color: #34e89e; }}
        .risk-medium {{ background: #4a4a1a; color: #ffd700; }}
        .risk-high {{ background: #4a1a1a; color: #e94560; }}
        .context-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }}
        .context-item {{
            background: #1e2a4a;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
        }}
        .footer {{
            background: #151d30;
            padding: 4px;
            text-align: center;
        }}
        .footer p {{
            margin: 4px 0;
            font-size: 11px;
            color: #5a6a8a;
        }}
        .footer .warning {{
            color: #ffd700;
            font-weight: 500;
        }}
        .text-green {{ color: #34e89e; }}
        .text-red {{ color: #e94560; }}
        .text-yellow {{ color: #ffd700; }}
        .text-blue {{ color: #4dabf7; }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 10px;
        }}
        .info-item {{
            background: #1e2a4a;
            padding: 8px;
            margin-bottom: 4px;
            border-radius: 6px;
            text-align: center;
        }}
        .info-label {{ font-size: 10px; color: #8892b0; }}
        .info-value {{ font-size: 13px; font-weight: 600; }}
        @media (max-width: 600px) {{
            .indicator-grid, .info-grid, .context-grid {{ grid-template-columns: 1fr; }}
            .price-value {{ font-size: 24px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Crypto Trading Bot</h1>
            <div class="subtitle">{now}</div>
            <div class="mode-badge">{mode_icon} {mode_text} Analysis</div>
        </div>
        <div class="main-content">
{html_content}
        </div>
        <div class="footer">
            <p class="warning">⚠️ This is automated analysis. Always DYOR.</p>
            <p>Crypto Trading Bot v2.1</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _parse_and_convert(self, text):
        """Parse console output และแปลงเป็น structured HTML"""
        import html as html_lib
        text = html_lib.escape(text)
        html_parts = []

        # === Extract Data ===

        # Price
        price_match = re.search(r"ราคาปัจจุบัน: \$([0-9,]+\.[0-9]+)", text)
        current_price = price_match.group(1) if price_match else None

        # Symbol & Date
        symbol_match = re.search(r"STRATEGY - (\w+)", text)
        symbol = symbol_match.group(1) if symbol_match else "BTCUSDT"

        date_match = re.search(r"วันที่: ([0-9-]+ [0-9:]+)", text)
        date_str = date_match.group(1) if date_match else ""

        # Leverage
        leverage_match = re.search(r"Leverage: (\d+)x", text)
        leverage = leverage_match.group(1) if leverage_match else "5"

        # Signal percentages - Updated regex
        long_match = re.search(r"LONG Signals: (\d+) \(([0-9.]+)%\)", text)
        short_match = re.search(r"SHORT Signals: (\d+) \(([0-9.]+)%\)", text)
        neutral_match = re.search(r"NEUTRAL Signals: (\d+) \(([0-9.]+)%\)", text)

        long_count = int(long_match.group(1)) if long_match else 0
        long_pct = float(long_match.group(2)) if long_match else 0
        short_count = int(short_match.group(1)) if short_match else 0
        short_pct = float(short_match.group(2)) if short_match else 0
        neutral_count = int(neutral_match.group(1)) if neutral_match else 0
        neutral_pct = float(neutral_match.group(2)) if neutral_match else 0

        # Recommendation - Updated regex
        recommendation = "WAIT"
        confidence = 0
        confidence_match = re.search(r"(?:STRONG |MODERATE )?(LONG|SHORT) SIGNAL \(([0-9.]+)%\)", text)
        if confidence_match:
            recommendation = confidence_match.group(1)
            confidence = float(confidence_match.group(2))
        else:
            confidence = max(long_pct, short_pct)

        # Trade Setup - Updated regex for new format
        entry_match = re.search(r"Entry: \$([0-9,]+\.[0-9]+)", text)
        sl_match = re.search(r"Stop Loss: \$([0-9,]+\.[0-9]+) \(([^)]+)\)", text)
        tp1_match = re.search(r"TP1 \(40%\): \$([0-9,]+\.[0-9]+) \(([^)]+)\)", text)
        tp2_match = re.search(r"TP2 \(30%\): \$([0-9,]+\.[0-9]+) \(([^)]+)\)", text)
        tp3_match = re.search(r"TP3 \(30%\): \$([0-9,]+\.[0-9]+) \(([^)]+)\)", text)

        # Position Management - Updated regex
        risk_match = re.search(r"Risk per Trade: ([0-9.]+)% \(\$([0-9,]+\.[0-9]+)\)", text)
        margin_match = re.search(r"Margin Required: \$([0-9,]+\.[0-9]+)", text)
        position_match = re.search(r"Position Size: \$([0-9,]+\.[0-9]+)", text)

        # Risk/Reward
        rr1_match = re.search(r"TP1: 1:([0-9.]+)", text)
        rr2_match = re.search(r"TP2: 1:([0-9.]+)", text)
        rr3_match = re.search(r"TP3: 1:([0-9.]+)", text)

        # ATR
        atr_match = re.search(r"ATR: \$([0-9,]+\.[0-9]+) \(([0-9.]+)%\)", text)

        # Support/Resistance
        support_matches = re.findall(r"S(\d): \$([0-9,]+\.[0-9]+)", text)
        resistance_matches = re.findall(r"R(\d): \$([0-9,]+\.[0-9]+)", text)

        # Market Context (New)
        regime_match = re.search(r"Regime: ([A-Z_]+) \(([0-9]+)% confidence\)", text)
        adx_bb_match = re.search(r"ADX: ([0-9.]+) \| BB Width: ([0-9.]+)%", text)
        price_range_match = re.search(r"Price Range \(20d\): ([0-9.]+)%", text)

        # Multi-Indicator Confirmation (New)
        multi_ind_match = re.search(r"Direction: (bullish|bearish|neutral) \((\d+)/6 indicators\)", text)
        confirm_match = re.search(r"Confirmation: ([0-9]+)% \| Strength: ([0-9]+)%", text)

        # Risk Assessment (New)
        risk_score_match = re.search(r"Risk Score: [🟢🟡🔴] (\d+)/100 \(([A-Z]+)\)", text)

        # Volatility Analysis (New)
        vol_current_match = re.search(r"Current ATR: ([0-9.]+)% \| Avg: ([0-9.]+)%", text)
        vol_ratio_match = re.search(r"Volatility Ratio: [🔥📉⚖️] ([0-9.]+)x", text)
        risk_adj_match = re.search(r"Risk Adjustment: (.+?)(?:\n|$)", text)

        # Candlestick Patterns (New)
        candle_bullish = re.search(r"Bullish: ([^\n]+)", text)
        candle_bearish = re.search(r"Bearish: ([^\n]+)", text)
        candle_score_match = re.search(r"Net Score: ([+-]?\d+)", text)

        # Trend Scores
        weekly_trend_match = re.search(r"Weekly Trend Score: [🟢🔴🟡] ([+-]?\d+)", text)
        daily_trend_match = re.search(r"Daily Trend Score: [🟢🔴🟡] ([+-]?\d+)", text)

        # Indicators - Updated patterns
        indicators = []

        # Weekly/Daily indicators
        rsi_matches = re.findall(r"RSI: ([0-9.]+)", text)
        if rsi_matches:
            rsi = float(rsi_matches[-1])  # Use daily RSI
            rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "yellow"
            indicators.append(("RSI", f"{rsi:.1f}", rsi_color))

        macd_matches = re.findall(r"MACD: ([0-9.-]+)", text)
        if macd_matches:
            macd = float(macd_matches[-1])
            macd_color = "green" if macd > 0 else "red"
            indicators.append(("MACD", f"{macd:.0f}", macd_color))

        adx_match = re.search(r"ADX: ([0-9.]+) \(DI\+: ([0-9.]+), DI-: ([0-9.]+)\)", text)
        if adx_match:
            adx = float(adx_match.group(1))
            adx_color = "green" if adx > 25 else "yellow"
            indicators.append(("ADX", f"{adx:.1f}", adx_color))

        if atr_match:
            indicators.append(("ATR", f"{atr_match.group(2)}%", "blue"))

        mfi_match = re.search(r"MFI: ([0-9.]+)", text)
        if mfi_match:
            mfi = float(mfi_match.group(1))
            mfi_color = "green" if mfi < 20 else "red" if mfi > 80 else "yellow"
            indicators.append(("MFI", f"{mfi:.1f}", mfi_color))

        cci_match = re.search(r"CCI: ([0-9.-]+)", text)
        if cci_match:
            cci = float(cci_match.group(1))
            cci_color = "green" if cci < -100 else "red" if cci > 100 else "yellow"
            indicators.append(("CCI", f"{cci:.0f}", cci_color))

        # Supertrend
        supertrend_match = re.search(r"Supertrend: (Bullish|Bearish)", text)

        # === Build HTML ===

        # Price Display
        if current_price:
            html_parts.append(f"""
            <div class="price-display">
                <div class="price-label">{symbol} Current Price</div>
                <div class="price-value">${current_price}</div>
                <div style="font-size: 11px; color: #8892b0; margin-top: 4px;">{date_str}</div>
            </div>
            """)

        # Signal Box
        signal_class = "signal-long" if recommendation == "LONG" else "signal-short" if recommendation == "SHORT" else "signal-wait"
        signal_icon = "✅" if recommendation == "LONG" else "❌" if recommendation == "SHORT" else "⏸️"
        signal_text = f"{recommendation} SIGNAL" if recommendation != "WAIT" else "WAIT - No Clear Signal"

        html_parts.append(f"""
        <div class="signal-box {signal_class}">
            <div class="signal-title">{signal_icon} {signal_text}</div>
            <div class="signal-subtitle">Confidence: {confidence:.1f}% | Leverage: {leverage}x</div>
            <div class="progress-bar">
                <div class="progress-long" style="width: {long_pct}%"></div>
                <div class="progress-short" style="width: {short_pct}%"></div>
                <div style="background: #6b6b2a; height: 100%; width: {neutral_pct}%"></div>
            </div>
            <div class="progress-labels">
                <span class="text-green">🟢 Long {long_count} ({long_pct:.1f}%)</span>
                <span style="color: #ffd700;">⚪ Neutral {neutral_count} ({neutral_pct:.1f}%)</span>
                <span class="text-red">🔴 Short {short_count} ({short_pct:.1f}%)</span>
            </div>
        </div>
        """)

        # Market Context (New Section)
        if regime_match:
            regime = regime_match.group(1)
            regime_conf = regime_match.group(2)
            adx_val = adx_bb_match.group(1) if adx_bb_match else "-"
            bb_width = adx_bb_match.group(2) if adx_bb_match else "-"
            price_range = price_range_match.group(1) if price_range_match else "-"

            # Color based on regime
            regime_color = "green" if "UPTREND" in regime else "red" if "DOWNTREND" in regime else "yellow"

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">🌍 Market Context</div>
                <div class="context-grid">
                    <div class="context-item">
                        <div class="info-label">Regime</div>
                        <div class="info-value text-{regime_color}">{regime.replace('_', ' ')}</div>
                        <div style="font-size: 10px; color: #8892b0;">{regime_conf}% confidence</div>
                    </div>
                    <div class="context-item">
                        <div class="info-label">ADX / BB Width</div>
                        <div class="info-value">{adx_val} / {bb_width}%</div>
                    </div>
                    <div class="context-item">
                        <div class="info-label">Price Range (20d)</div>
                        <div class="info-value">{price_range}%</div>
                    </div>
                </div>
            </div>
            """)

        # Multi-Indicator Confirmation (New Section)
        if multi_ind_match:
            direction = multi_ind_match.group(1).upper()
            ind_count = multi_ind_match.group(2)
            confirm_pct = confirm_match.group(1) if confirm_match else "0"
            strength = confirm_match.group(2) if confirm_match else "0"
            dir_color = "green" if direction == "BULLISH" else "red" if direction == "BEARISH" else "yellow"

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">🎯 Multi-Indicator Confirmation</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Direction</div>
                        <div class="info-value text-{dir_color}">{direction} ({ind_count}/6)</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Confirmation</div>
                        <div class="info-value">{confirm_pct}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Strength</div>
                        <div class="info-value">{strength}%</div>
                    </div>
                </div>
            </div>
            """)

        # Risk Assessment (New Section)
        if risk_score_match:
            risk_score = risk_score_match.group(1)
            risk_level = risk_score_match.group(2)
            risk_class = "risk-low" if risk_level == "LOW" else "risk-high" if risk_level == "HIGH" else "risk-medium"

            # Extract risk factors
            risk_factors = re.findall(r"    - ([^\n]+)", text)

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">⚠️ Risk Assessment</div>
                <div style="text-align: center; margin-bottom: 12px;">
                    <span class="risk-badge {risk_class}">{risk_level} RISK</span>
                    <div style="font-size: 24px; font-weight: 700; margin-top: 8px;">{risk_score}/100</div>
                </div>
            """)

            if risk_factors:
                html_parts.append('<ul class="signal-list">')
                for factor in risk_factors[:5]:
                    html_parts.append(f'<li class="neutral">{factor}</li>')
                html_parts.append('</ul>')

            html_parts.append('</div>')

        # Volatility Analysis (New Section)
        if vol_current_match:
            current_atr = vol_current_match.group(1)
            avg_atr = vol_current_match.group(2)
            vol_ratio = vol_ratio_match.group(1) if vol_ratio_match else "1.0"
            risk_note = risk_adj_match.group(1) if risk_adj_match else ""

            vol_color = "red" if float(vol_ratio) > 1.5 else "green" if float(vol_ratio) < 0.8 else "yellow"

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">📊 Volatility Analysis</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Current ATR</div>
                        <div class="info-value">{current_atr}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Average ATR</div>
                        <div class="info-value">{avg_atr}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Volatility Ratio</div>
                        <div class="info-value text-{vol_color}">{vol_ratio}x</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Risk Adjustment</div>
                        <div class="info-value" style="font-size: 11px;">{risk_note[:30]}</div>
                    </div>
                </div>
            </div>
            """)

        # Trade Setup
        if entry_match:
            entry = entry_match.group(1)
            sl = sl_match.group(1) if sl_match else "-"
            sl_info = sl_match.group(2) if sl_match else ""
            tp1 = tp1_match.group(1) if tp1_match else "-"
            tp1_info = tp1_match.group(2) if tp1_match else ""
            tp2 = tp2_match.group(1) if tp2_match else "-"
            tp2_info = tp2_match.group(2) if tp2_match else ""
            tp3 = tp3_match.group(1) if tp3_match else "-"
            tp3_info = tp3_match.group(2) if tp3_match else ""

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">💼 Trade Setup</div>
                <div class="trade-setup">
                    <div class="trade-row entry">
                        <span class="trade-label">🎯 Entry</span>
                        <span class="trade-value yellow">${entry}</span>
                    </div>
                    <div class="trade-row sl">
                        <span class="trade-label">🛡️ Stop Loss</span>
                        <span class="trade-value red">${sl} <small style="color:#8892b0">({sl_info})</small></span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 TP1 (40%)</span>
                        <span class="trade-value green">${tp1} <small style="color:#8892b0">({tp1_info})</small></span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 TP2 (30%)</span>
                        <span class="trade-value green">${tp2} <small style="color:#8892b0">({tp2_info})</small></span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 TP3 (30%)</span>
                        <span class="trade-value green">${tp3} <small style="color:#8892b0">({tp3_info})</small></span>
                    </div>
                </div>
            </div>
            """)

        # Position & Risk Management
        if risk_match or margin_match:
            risk_pct = risk_match.group(1) if risk_match else "2"
            risk_amt = risk_match.group(2) if risk_match else "-"
            margin = margin_match.group(1) if margin_match else "-"
            pos_size = position_match.group(1) if position_match else "-"
            rr1 = rr1_match.group(1) if rr1_match else "-"
            rr2 = rr2_match.group(1) if rr2_match else "-"
            rr3 = rr3_match.group(1) if rr3_match else "-"

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">💰 Position Management</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Risk</div>
                        <div class="info-value text-yellow">{risk_pct}% (${risk_amt})</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Margin</div>
                        <div class="info-value">${margin}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Position Size</div>
                        <div class="info-value text-blue">${pos_size}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">R:R Ratio</div>
                        <div class="info-value text-green">1:{rr1} / 1:{rr2} / 1:{rr3}</div>
                    </div>
                </div>
            </div>
            """)

        # Key Indicators
        if indicators:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">📈 Key Indicators</div>')
            html_parts.append('<div class="indicator-grid">')
            for label, value, color in indicators:
                html_parts.append(f"""
                <div class="indicator-item">
                    <div class="indicator-label">{label}</div>
                    <div class="indicator-value text-{color}">{value}</div>
                </div>
                """)

            # Add Trend Scores
            if weekly_trend_match:
                wt = int(weekly_trend_match.group(1))
                wt_color = "green" if wt > 0 else "red" if wt < 0 else "yellow"
                html_parts.append(f"""
                <div class="indicator-item">
                    <div class="indicator-label">Weekly Trend</div>
                    <div class="indicator-value text-{wt_color}">{wt:+d}</div>
                </div>
                """)

            if daily_trend_match:
                dt = int(daily_trend_match.group(1))
                dt_color = "green" if dt > 0 else "red" if dt < 0 else "yellow"
                html_parts.append(f"""
                <div class="indicator-item">
                    <div class="indicator-label">Daily Trend</div>
                    <div class="indicator-value text-{dt_color}">{dt:+d}</div>
                </div>
                """)

            if supertrend_match:
                st = supertrend_match.group(1)
                st_color = "green" if st == "Bullish" else "red"
                html_parts.append(f"""
                <div class="indicator-item">
                    <div class="indicator-label">Supertrend</div>
                    <div class="indicator-value text-{st_color}">{st}</div>
                </div>
                """)

            html_parts.append('</div>')
            html_parts.append('</div>')

        # Candlestick Patterns (New Section)
        if candle_bullish or candle_bearish:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">🕯️ Candlestick Patterns</div>')

            candle_net = int(candle_score_match.group(1)) if candle_score_match else 0
            score_color = "green" if candle_net > 0 else "red" if candle_net < 0 else "yellow"

            html_parts.append(f'<div style="text-align: center; margin-bottom: 12px;">Net Score: <span class="text-{score_color}" style="font-size: 18px; font-weight: 700;">{candle_net:+d}</span></div>')

            html_parts.append('<div class="info-grid">')
            if candle_bullish:
                patterns = candle_bullish.group(1).strip()
                html_parts.append(f"""
                <div class="info-item">
                    <div class="info-label">🟢 Bullish</div>
                    <div class="info-value text-green" style="font-size: 11px;">{patterns}</div>
                </div>
                """)
            if candle_bearish:
                patterns = candle_bearish.group(1).strip()
                html_parts.append(f"""
                <div class="info-item">
                    <div class="info-label">🔴 Bearish</div>
                    <div class="info-value text-red" style="font-size: 11px;">{patterns}</div>
                </div>
                """)
            html_parts.append('</div>')
            html_parts.append('</div>')

        # Support & Resistance
        if support_matches or resistance_matches:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">📊 Support & Resistance</div>')

            html_parts.append('<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">')

            # Resistance
            html_parts.append('<div>')
            html_parts.append('<div style="font-size: 11px; color: #e94560; margin-bottom: 6px;">🔒 RESISTANCE</div>')
            html_parts.append('<ul class="level-list">')
            for r in resistance_matches[:3]:
                html_parts.append(f'<li><span>R{r[0]}</span><span class="text-red">${r[1]}</span></li>')
            html_parts.append('</ul>')
            html_parts.append('</div>')

            # Support
            html_parts.append('<div>')
            html_parts.append('<div style="font-size: 11px; color: #34e89e; margin-bottom: 6px;">🛡️ SUPPORT</div>')
            html_parts.append('<ul class="level-list">')
            for s in support_matches[:3]:
                html_parts.append(f'<li><span>S{s[0]}</span><span class="text-green">${s[1]}</span></li>')
            html_parts.append('</ul>')
            html_parts.append('</div>')

            html_parts.append('</div>')
            html_parts.append('</div>')

        # Signal Details
        long_signals = re.findall(r"  ((?:📈|📊|💪|💰|✅|🔥|🚀|🔄|☁️|🎯|⬆️)[^\n]+)", text.split("SHORT Signals")[0]) if "SHORT Signals" in text else []
        short_section = text.split("SHORT Signals")[-1].split("NEUTRAL Signals")[0] if "SHORT Signals" in text else ""
        short_signals = re.findall(r"  ((?:📉|📊|💪|⚠️|❌|🔥|🔻|🔄|☁️|🎯|⬇️)[^\n]+)", short_section)

        if long_signals or short_signals:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">📋 Signal Details</div>')
            html_parts.append('<ul class="signal-list">')

            for sig in long_signals[:8]:
                html_parts.append(f'<li class="long">{sig.strip()}</li>')

            for sig in short_signals[:8]:
                html_parts.append(f'<li class="short">{sig.strip()}</li>')

            html_parts.append('</ul>')
            html_parts.append('</div>')

        return "\n".join(html_parts)

    def send_email(self, console_output, mode):
        """ส่งอีเมล"""
        try:
            mode_icon = "📅" if mode.lower() == "weekly" else "🌙"
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{mode_icon} Crypto Bot - {mode.upper()} Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email

            # Text version
            text_part = MIMEText(console_output, "plain", "utf-8")
            msg.attach(text_part)

            # HTML version
            html_content = self.create_html_email(console_output, mode)
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(html_part)

            # ส่งอีเมล
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            print("\n" + "=" * 50)
            print(f"📧 {mode.upper()} EMAIL SENT SUCCESSFULLY!")
            print(f"   To: {self.recipient_email}")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"\n❌ Failed to send email: {e}")
            print("💡 Make sure you have set up App Password for Gmail")
            return False

    def create_deepseek_html_email(self, analysis_result: dict, market_summary: dict):
        """สร้าง HTML email สำหรับ DeepSeek AI Analysis"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert markdown to HTML
        analysis_text = analysis_result.get("analysis", "No analysis available")
        analysis_html = markdown.markdown(
            analysis_text,
            extensions=["tables", "fenced_code", "nl2br"]
        )

        # Extract key metrics from market summary
        current_price = market_summary.get("current_price", 0)
        swing_data = market_summary.get("swing", {})
        monthly_data = market_summary.get("monthly", {})

        # Get signal summaries
        swing_signals = swing_data.get("signals", {})
        monthly_signals = monthly_data.get("signals", {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #e8e8e8;
            padding: 2px;
            margin: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            color: #fff;
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }}
        .header .subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            margin-top: 8px;
        }}
        .header .ai-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 6px 20px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 12px;
        }}
        .price-banner {{
            background: linear-gradient(135deg, #1e2a4a 0%, #2d3a5a 100%);
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #3a4a6b;
        }}
        .price-value {{
            font-size: 36px;
            font-weight: 700;
            color: #ffd700;
        }}
        .price-label {{
            font-size: 12px;
            color: #8892b0;
            text-transform: uppercase;
        }}
        .signals-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 15px;
            background: #1e2a4a;
        }}
        .signal-card {{
            background: #253557;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .signal-card h3 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #8892b0;
        }}
        .signal-bar {{
            height: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            margin: 8px 0;
        }}
        .bar-long {{ background: #34e89e; height: 100%; }}
        .bar-short {{ background: #e94560; height: 100%; }}
        .signal-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
        }}
        .main-content {{
            padding: 20px;
        }}
        .ai-analysis {{
            background: #253557;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .ai-analysis h2 {{
            color: #667eea;
            font-size: 18px;
            margin: 0 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3a4a6b;
        }}
        .ai-analysis h3 {{
            color: #4dabf7;
            font-size: 14px;
            margin: 20px 0 10px 0;
        }}
        .ai-analysis p {{
            margin: 8px 0;
            font-size: 13px;
            color: #c8d0e8;
        }}
        .ai-analysis ul, .ai-analysis ol {{
            margin: 8px 0;
            padding-left: 20px;
        }}
        .ai-analysis li {{
            margin: 5px 0;
            font-size: 13px;
            color: #c8d0e8;
        }}
        .ai-analysis strong {{
            color: #ffd700;
        }}
        .ai-analysis code {{
            background: #1a1a2e;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            color: #4dabf7;
        }}
        .footer {{
            background: #0f0c29;
            padding: 15px;
            text-align: center;
        }}
        .footer p {{
            margin: 4px 0;
            font-size: 11px;
            color: #5a6a8a;
        }}
        .footer .powered {{
            color: #667eea;
            font-weight: 600;
        }}
        .footer .warning {{
            color: #ffd700;
        }}
        .text-green {{ color: #34e89e; }}
        .text-red {{ color: #e94560; }}
        .text-yellow {{ color: #ffd700; }}
        @media (max-width: 600px) {{
            .signals-grid {{ grid-template-columns: 1fr; }}
            .price-value {{ font-size: 28px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 DeepSeek AI Analysis</h1>
            <div class="subtitle">{now}</div>
            <div class="ai-badge">🧠 Powered by DeepSeek AI</div>
        </div>

        <div class="price-banner">
            <div class="price-label">BTCUSDT Current Price</div>
            <div class="price-value">${current_price:,.2f}</div>
        </div>

        <div class="signals-grid">
            <div class="signal-card">
                <h3>🔄 Swing Trading (2-10 days)</h3>
                <div class="signal-bar">
                    <div class="bar-long" style="width: {swing_signals.get('long_pct', 0)}%"></div>
                    <div class="bar-short" style="width: {swing_signals.get('short_pct', 0)}%"></div>
                </div>
                <div class="signal-labels">
                    <span class="text-green">Long {swing_signals.get('long_pct', 0):.0f}%</span>
                    <span class="text-red">Short {swing_signals.get('short_pct', 0):.0f}%</span>
                </div>
            </div>
            <div class="signal-card">
                <h3>🌙 Monthly Trading</h3>
                <div class="signal-bar">
                    <div class="bar-long" style="width: {monthly_signals.get('long_pct', 0)}%"></div>
                    <div class="bar-short" style="width: {monthly_signals.get('short_pct', 0)}%"></div>
                </div>
                <div class="signal-labels">
                    <span class="text-green">Long {monthly_signals.get('long_pct', 0):.0f}%</span>
                    <span class="text-red">Short {monthly_signals.get('short_pct', 0):.0f}%</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="ai-analysis">
                <h2>🧠 AI Analysis Report</h2>
                {analysis_html}
            </div>
        </div>

        <div class="footer">
            <p class="warning">⚠️ This is AI-generated analysis. Always DYOR.</p>
            <p class="powered">Powered by DeepSeek AI | Model: {analysis_result.get('model', 'deepseek-chat')}</p>
            <p>Crypto Trading Bot v2.1</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_deepseek_email(self, analysis_result: dict, market_summary: dict):
        """ส่งอีเมล DeepSeek AI Analysis"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🤖 DeepSeek AI Analysis - BTC ${market_summary.get('current_price', 0):,.0f} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email

            # Text version (plain markdown)
            text_content = f"""
DeepSeek AI Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current BTC Price: ${market_summary.get('current_price', 0):,.2f}

{analysis_result.get('analysis', 'No analysis available')}

---
Powered by DeepSeek AI
⚠️ This is AI-generated analysis. Always DYOR.
"""
            text_part = MIMEText(text_content, "plain", "utf-8")
            msg.attach(text_part)

            # HTML version
            html_content = self.create_deepseek_html_email(analysis_result, market_summary)
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            print("\n" + "=" * 50)
            print("🤖 DEEPSEEK AI ANALYSIS EMAIL SENT!")
            print(f"   To: {self.recipient_email}")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"\n❌ Failed to send DeepSeek email: {e}")
            return False

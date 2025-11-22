"""
Email Notifier Module
ส่งอีเมลแจ้งเตือนผลการวิเคราะห์ Trading Bot
"""

import re
import smtplib
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

        # Parse และแปลง console output เป็น structured HTML
        html_content = self._parse_and_convert(console_output)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            padding: 4px;
            margin: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 700px;
            margin: 0 auto;
            background-color: #1e2a4a;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #e94560 0%, #c73e54 100%);
            padding: 4px;
            text-align: center;
        }}
        .header h1 {{
            color: #fff;
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .header .subtitle {{
            color: rgba(255,255,255,0.85);
            font-size: 14px;
            margin-top: 8px;
        }}
        .header .mode-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Main Content */
        .main-content {{
            padding: 4px;
        }}

        /* Section Cards */
        .section {{
            background: #253557;
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 16px;
        }}
        .section-title {{
            font-size: 16px;
            font-weight: 700;
            color: #4dabf7;
            margin: 0 0 16px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3a4a6b;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Price Display */
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

        /* Signal Box */
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
        .signal-subtitle {{
            font-size: 14px;
            opacity: 0.9;
        }}

        /* Indicator Grid */
        .indicator-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}
        .indicator-item {{
            background: #1e2a4a;
            border-radius: 8px;
            padding: 12px;
            border-left: 3px solid #4dabf7;
        }}
        .indicator-label {{
            font-size: 11px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .indicator-value {{
            font-size: 16px;
            font-weight: 600;
            color: #e8e8e8;
            margin-top: 4px;
        }}

        /* Signal List */
        .signal-list {{
            margin: 0;
            padding: 0;
            list-style: none;
        }}
        .signal-list li {{
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
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

        /* Trade Setup */
        .trade-setup {{
            background: #1e2a4a;
            border-radius: 8px;
            overflow: hidden;
        }}
        .trade-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid #3a4a6b;
        }}
        .trade-row:last-child {{
            border-bottom: none;
        }}
        .trade-row.entry {{
            background: #2a3a5a;
        }}
        .trade-row.sl {{
            background: rgba(233, 69, 96, 0.1);
        }}
        .trade-row.tp {{
            background: rgba(52, 232, 158, 0.1);
        }}
        .trade-label {{
            font-size: 13px;
            color: #8892b0;
        }}
        .trade-value {{
            font-size: 14px;
            font-weight: 600;
        }}
        .trade-value.green {{ color: #34e89e; }}
        .trade-value.red {{ color: #e94560; }}
        .trade-value.yellow {{ color: #ffd700; }}

        /* Progress Bar */
        .signal-progress {{
            margin-top: 16px;
        }}
        .progress-bar {{
            height: 8px;
            background: #1e2a4a;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
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
            margin-top: 8px;
            font-size: 12px;
        }}

        /* Footer */
        .footer {{
            background: #151d30;
            padding: 4px;
            text-align: center;
        }}
        .footer p {{
            margin: 0;
            font-size: 12px;
            color: #5a6a8a;
        }}
        .footer .warning {{
            color: #ffd700;
            font-weight: 500;
            margin-bottom: 8px;
        }}

        /* Raw Content (fallback) */
        .raw-content {{
            background: #0d1526;
            border-radius: 8px;
            padding: 4px;
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 12px;
            line-height: 1.8;
            white-space: pre-wrap;
            overflow-x: auto;
            color: #a8b2c4;
        }}

        /* Colors */
        .text-green {{ color: #34e89e; }}
        .text-red {{ color: #e94560; }}
        .text-yellow {{ color: #ffd700; }}
        .text-blue {{ color: #4dabf7; }}
        .text-muted {{ color: #8892b0; }}

        /* Responsive */
        @media (max-width: 600px) {{
            .indicator-grid {{
                grid-template-columns: 1fr;
            }}
            .price-value {{
                font-size: 24px;
            }}
            .header h1 {{
                font-size: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Crypto Trading Bot 🔥</h1>
            <div class="subtitle">Analysis Report • {now}</div>
            <div class="mode-badge">📊 {mode.upper()} Analysis</div>
        </div>

        <div class="main-content">
{html_content}
        </div>

        <div class="footer">
            <p class="warning">⚠️ This is an automated analysis. Always do your own research.</p>
            <p>Generated by Crypto Trading Bot</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _parse_and_convert(self, text):
        """Parse console output และแปลงเป็น structured HTML"""
        import html as html_lib

        # Escape HTML characters
        text = html_lib.escape(text)

        # Extract key information using regex
        html_parts = []

        # Find current price
        price_match = re.search(r"ราคาปัจจุบัน: \$([0-9,]+\.[0-9]+)", text)
        current_price = price_match.group(1) if price_match else None

        # Find signal percentages
        long_match = re.search(r"LONG Signals: (\d+) \(([0-9.]+)%\)", text)
        short_match = re.search(r"SHORT Signals: (\d+) \(([0-9.]+)%\)", text)

        long_pct = float(long_match.group(2)) if long_match else 0
        short_pct = float(short_match.group(2)) if short_match else 0

        # Find recommendation
        recommendation = "WAIT"
        if "LONG SIGNAL" in text:
            recommendation = "LONG"
        elif "SHORT SIGNAL" in text:
            recommendation = "SHORT"

        # Find trade setup
        entry_match = re.search(r"Entry: \$([0-9,]+\.[0-9]+)", text)
        sl_match = re.search(r"Stop Loss: \$([0-9,]+\.[0-9]+)", text)
        tp1_match = re.search(r"TP1.*?: \$([0-9,]+\.[0-9]+)", text)
        tp2_match = re.search(r"TP2.*?: \$([0-9,]+\.[0-9]+)", text)
        tp3_match = re.search(r"TP3.*?: \$([0-9,]+\.[0-9]+)", text)

        # Price Display
        if current_price:
            html_parts.append(f"""
            <div class="price-display">
                <div class="price-label">Current Price</div>
                <div class="price-value">${current_price}</div>
            </div>
            """)

        # Signal Summary with Progress Bar
        if long_match and short_match:
            signal_class = "signal-long" if recommendation == "LONG" else "signal-short" if recommendation == "SHORT" else "signal-wait"
            signal_icon = "✅" if recommendation == "LONG" else "❌" if recommendation == "SHORT" else "⏸️"
            signal_text = "LONG SIGNAL" if recommendation == "LONG" else "SHORT SIGNAL" if recommendation == "SHORT" else "WAIT - No Clear Signal"

            html_parts.append(f"""
            <div class="signal-box {signal_class}">
                <div class="signal-title">{signal_icon} {signal_text}</div>
                <div class="signal-subtitle">Confidence: {max(long_pct, short_pct):.1f}%</div>
                <div class="signal-progress">
                    <div class="progress-bar">
                        <div class="progress-long" style="width: {long_pct}%"></div>
                        <div class="progress-short" style="width: {short_pct}%"></div>
                    </div>
                    <div class="progress-labels">
                        <span class="text-green">🟢 Long {long_pct:.1f}%</span>
                        <span class="text-red">🔴 Short {short_pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """)

        # Trade Setup
        if entry_match:
            entry = entry_match.group(1)
            sl = sl_match.group(1) if sl_match else "-"
            tp1 = tp1_match.group(1) if tp1_match else "-"
            tp2 = tp2_match.group(1) if tp2_match else "-"
            tp3 = tp3_match.group(1) if tp3_match else "-"

            html_parts.append(f"""
            <div class="section">
                <div class="section-title">💼 Trade Setup</div>
                <div class="trade-setup">
                    <div class="trade-row entry">
                        <span class="trade-label">🎯 Entry Price</span>
                        <span class="trade-value yellow">${entry}</span>
                    </div>
                    <div class="trade-row sl">
                        <span class="trade-label">🛡️ Stop Loss</span>
                        <span class="trade-value red">${sl}</span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 Take Profit 1 (40%)</span>
                        <span class="trade-value green">${tp1}</span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 Take Profit 2 (30%)</span>
                        <span class="trade-value green">${tp2}</span>
                    </div>
                    <div class="trade-row tp">
                        <span class="trade-label">🎁 Take Profit 3 (30%)</span>
                        <span class="trade-value green">${tp3}</span>
                    </div>
                </div>
            </div>
            """)

        # Extract Long Signals
        long_signals = re.findall(r"(?:📈|📊|💪|💰|✅|🔥|🚀|🔄|☁️)[^\n]+", text.split("SHORT Signals")[0]) if "SHORT Signals" in text else []

        # Extract Short Signals
        short_section = text.split("SHORT Signals")[-1].split("NEUTRAL Signals")[0] if "SHORT Signals" in text else ""
        short_signals = re.findall(r"(?:📉|📊|💪|⚠️|❌|🔥|🔻|🔄|☁️)[^\n]+", short_section)

        # Signal Details
        if long_signals or short_signals:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">📊 Signal Details</div>')
            html_parts.append('<ul class="signal-list">')

            for sig in long_signals[:6]:
                html_parts.append(f'<li class="long">{sig.strip()}</li>')

            for sig in short_signals[:6]:
                html_parts.append(f'<li class="short">{sig.strip()}</li>')

            html_parts.append('</ul>')
            html_parts.append('</div>')

        # Extract Indicators
        indicators = []

        rsi_match = re.search(r"RSI: ([0-9.]+)", text)
        if rsi_match:
            rsi = float(rsi_match.group(1))
            rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "yellow"
            indicators.append(("RSI", f"{rsi:.1f}", rsi_color))

        macd_match = re.search(r"MACD: ([0-9.-]+)", text)
        if macd_match:
            macd = float(macd_match.group(1))
            macd_color = "green" if macd > 0 else "red"
            indicators.append(("MACD", f"{macd:.2f}", macd_color))

        adx_match = re.search(r"ADX: ([0-9.]+)", text)
        if adx_match:
            indicators.append(("ADX", adx_match.group(1), "blue"))

        atr_match = re.search(r"ATR: \$([0-9,]+\.[0-9]+) \(([0-9.]+)%\)", text)
        if atr_match:
            indicators.append(("ATR", f"${atr_match.group(1)} ({atr_match.group(2)}%)", "blue"))

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
            html_parts.append('</div>')
            html_parts.append('</div>')

        # If no structured content was found, show raw content
        if not html_parts:
            styled_text = self._style_raw_text(text)
            html_parts.append(f'<div class="raw-content">{styled_text}</div>')

        return "\n".join(html_parts)

    def _style_raw_text(self, text):
        """Style raw text with colors"""
        replacements = [
            ("✅", '<span class="text-green">✅</span>'),
            ("❌", '<span class="text-red">❌</span>'),
            ("⚠️", '<span class="text-yellow">⚠️</span>'),
            ("🟢", '<span class="text-green">🟢</span>'),
            ("🔴", '<span class="text-red">🔴</span>'),
            ("🟡", '<span class="text-yellow">🟡</span>'),
            ("📈", '<span class="text-green">📈</span>'),
            ("📉", '<span class="text-red">📉</span>'),
            ("💰", '<span class="text-yellow">💰</span>'),
            ("🎯", '<span class="text-blue">🎯</span>'),
            ("🛡️", '<span class="text-green">🛡️</span>'),
            ("🎁", '<span class="text-yellow">🎁</span>'),
            ("💵", '<span class="text-green">💵</span>'),
            ("🔥", '<span class="text-red">🔥</span>'),
            ("📊", '<span class="text-blue">📊</span>'),
            ("💪", '<span class="text-green">💪</span>'),
            ("🚀", '<span class="text-green">🚀</span>'),
            ("🔻", '<span class="text-red">🔻</span>'),
            ("☁️", '<span class="text-blue">☁️</span>'),
            ("=" * 100, '<hr style="border-color: #3a4a6b; margin: 16px 0;">'),
            ("=" * 50, '<hr style="border-color: #3a4a6b; margin: 12px 0;">'),
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        return text

    def send_email(self, console_output, mode):
        """ส่งอีเมล"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🔥 Crypto Bot Alert - {mode.upper()} Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
            print("📧 EMAIL SENT SUCCESSFULLY!")
            print(f"   To: {self.recipient_email}")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"\n❌ Failed to send email: {e}")
            print("💡 Make sure you have set up App Password for Gmail")
            print("   1. Go to Google Account > Security")
            print("   2. Enable 2-Factor Authentication")
            print("   3. Create App Password for 'Mail'")
            return False

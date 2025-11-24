#!/usr/bin/env python3
"""
Crypto Trading Analysis Bot
Main entry point
"""

import sys
import os
import io

from dotenv import load_dotenv
from trading import SwingTradingStrategy, MonthlyTradingStrategy
from email_notifier import EmailNotifier
from deepseek_analyzer import DeepSeekAnalyzer


def print_banner():
    """แสดง banner ของ bot"""
    print("\n" + "🔥" * 50)
    print("          CRYPTO TRADING ANALYSIS BOT")
    print("          Enhanced with Advanced Indicators")
    print("🔥" * 50 + "\n")


def print_usage():
    """แสดงวิธีใช้งาน"""
    print("Usage: python main.py [swing|monthly|both|deepseek] [--no-email] [--no-ai]")
    print("  swing   (s)   - Show swing trading analysis only (2-10 days)")
    print("  monthly (m)   - Show monthly analysis only")
    print("  both          - Show both analyses (default)")
    print("  deepseek (ai) - Run DeepSeek AI analysis only")
    print("  --no-email    - Skip sending email notification")
    print("  --no-ai       - Skip DeepSeek AI analysis email")
    print("")
    print("Email Configuration (Environment Variables):")
    print("  BOT_EMAIL_SENDER    - Gmail address for sending")
    print("  BOT_EMAIL_PASSWORD  - Gmail App Password")
    print("  BOT_EMAIL_RECIPIENT - Recipient email address")
    print("  DEEPSEEK_API_KEY    - DeepSeek API key for AI analysis")


def run_swing_analysis(symbol="BTCUSDT", balance=10000):
    """รันการวิเคราะห์ Swing Trading และ return output + strategy object"""
    print("\n" + "━" * 100)
    print("                         🔄 SWING TRADING ANALYSIS (2-10 Days)")
    print("━" * 100 + "\n")

    swing_trader = SwingTradingStrategy(symbol=symbol, leverage=5)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    swing_trader.get_swing_recommendation(balance=balance)
    swing_output = buffer.getvalue()
    sys.stdout = old_stdout

    print(swing_output)
    return swing_output, swing_trader


def run_monthly_analysis(symbol="BTCUSDT", balance=10000):
    """รันการวิเคราะห์ Monthly และ return output + strategy object"""
    print("\n" + "━" * 100)
    print("                         🌙 MONTHLY ANALYSIS")
    print("━" * 100 + "\n")

    monthly_trader = MonthlyTradingStrategy(symbol=symbol, leverage=3)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    monthly_trader.get_monthly_recommendation(balance=balance)
    monthly_output = buffer.getvalue()
    sys.stdout = old_stdout

    print(monthly_output)
    return monthly_output, monthly_trader


def get_email_config():
    """ดึงค่า email configuration จาก environment variables"""
    email_sender = os.getenv("BOT_EMAIL_SENDER")
    email_password = os.getenv("BOT_EMAIL_PASSWORD")
    email_recipient = os.getenv("BOT_EMAIL_RECIPIENT")

    return email_sender, email_password, email_recipient


def send_email_notification(output, mode, email_sender, email_password, email_recipient):
    """ส่งอีเมลแจ้งเตือน"""
    if email_sender and email_password:
        notifier = EmailNotifier(email_sender, email_password, email_recipient)
        notifier.send_email(output, mode)
        return True
    return False


def run_deepseek_analysis(email_sender, email_password, email_recipient, symbol="BTCUSDT"):
    """รัน DeepSeek AI Analysis แบบ Standalone (ดึงข้อมูลจาก Binance โดยตรง)"""
    deepseek = DeepSeekAnalyzer()

    if not deepseek.is_configured():
        print("\n" + "=" * 50)
        print("🤖 DEEPSEEK AI NOT CONFIGURED")
        print("   Set DEEPSEEK_API_KEY environment variable")
        print("   to enable AI-powered analysis")
        print("=" * 50)
        return False

    print("\n🤖 Running DeepSeek AI Standalone Analysis...")
    print(f"   Symbol: {symbol}")

    # Call standalone analysis (fetches from Binance directly)
    analysis_result = deepseek.analyze_standalone(symbol)

    if not analysis_result:
        print("   ❌ DeepSeek analysis failed: No response")
        return False

    if not analysis_result.get("success"):
        print(f"   ❌ DeepSeek analysis failed: {analysis_result.get('error', 'Unknown error')}")
        return False

    print("   ✅ Analysis complete!")

    # Prepare market summary for email (from the fetched data)
    market_data = analysis_result.get("market_data", {})
    market_summary = {
        "current_price": market_data.get("current_price", 0),
        "swing": market_data.get("timeframes", {}).get("4h", {}),
        "monthly": market_data.get("timeframes", {}).get("1w", {}),
    }

    # Send email
    if email_sender and email_password:
        notifier = EmailNotifier(email_sender, email_password, email_recipient)
        return notifier.send_deepseek_email(analysis_result, market_summary)

    return True


def print_email_not_configured():
    """แสดงข้อความเมื่อยังไม่ได้ตั้งค่าอีเมล"""
    print("\n" + "=" * 50)
    print("📧 EMAIL NOT CONFIGURED")
    print("   To enable email notifications:")
    print("   1. Set BOT_EMAIL_SENDER environment variable")
    print("   2. Set BOT_EMAIL_PASSWORD (Gmail App Password)")
    print("   3. Optionally set BOT_EMAIL_RECIPIENT")
    print("")
    print("   Example:")
    print("     export BOT_EMAIL_SENDER='your@gmail.com'")
    print("     export BOT_EMAIL_PASSWORD='your-app-password'")
    print("=" * 50)


def main():
    """Main entry point"""
    load_dotenv()

    # Configuration
    symbol = "BTCUSDT"
    balance = 10000

    # Parse arguments
    args = sys.argv[1:]

    # Determine mode
    mode = "both"
    for arg in args:
        if arg.lower() in ["swing", "s", "monthly", "m", "both", "all", "deepseek", "ai"]:
            mode = arg.lower()
            break

    # Check for --no-email flag
    send_email = "--no-email" not in args

    # Check for --no-ai flag
    send_ai_email = "--no-ai" not in args

    # Check for help
    if "-h" in args or "--help" in args:
        print_usage()
        return

    # Print banner
    print_banner()

    # Validate mode
    if mode not in ["swing", "s", "monthly", "m", "both", "all", "deepseek", "ai"]:
        print_usage()
        return

    # Get email config
    email_sender, email_password, email_recipient = get_email_config()
    email_configured = bool(email_sender and email_password)

    # Track outputs and strategy objects for separate emails
    swing_output = None
    monthly_output = None
    swing_trader = None
    monthly_trader = None

    # Run analyses
    if mode in ["swing", "s", "both", "all"]:
        swing_output, swing_trader = run_swing_analysis(symbol, balance)

    if mode in ["monthly", "m", "both", "all"]:
        monthly_output, monthly_trader = run_monthly_analysis(symbol, balance)

    # If mode is deepseek/ai only, skip strategy analysis
    if mode in ["deepseek", "ai"]:
        print("\n🤖 Running DeepSeek AI Analysis Only...")
        if email_configured and send_email:
            run_deepseek_analysis(
                email_sender, email_password, email_recipient, symbol
            )
        else:
            print_email_not_configured()
        return

    # Send separate emails
    if send_email:
        if email_configured:
            # ส่งอีเมลแยก 2 ฉบับ
            if swing_output:
                print("\n📧 Sending SWING TRADING analysis email...")
                send_email_notification(
                    swing_output, "swing",
                    email_sender, email_password, email_recipient
                )

            if monthly_output:
                print("\n📧 Sending MONTHLY analysis email...")
                send_email_notification(
                    monthly_output, "monthly",
                    email_sender, email_password, email_recipient
                )

            # ส่งอีเมล DeepSeek AI Analysis (ฉบับที่ 3) - Standalone
            if send_ai_email:
                print("\n🤖 Sending DeepSeek AI analysis email...")
                run_deepseek_analysis(
                    email_sender, email_password, email_recipient, symbol
                )
        else:
            print_email_not_configured()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Crypto Trading Analysis Bot v2.0
Main entry point
"""

import sys
import os
import io

from strategies import WeeklyTradingStrategy, MonthlyTradingStrategy
from email_notifier import EmailNotifier


def print_banner():
    """แสดง banner ของ bot"""
    print("\n" + "🔥" * 50)
    print("          CRYPTO TRADING ANALYSIS BOT v2.0")
    print("          Enhanced with Advanced Indicators")
    print("🔥" * 50 + "\n")


def print_usage():
    """แสดงวิธีใช้งาน"""
    print("Usage: python main.py [weekly|monthly|both] [--no-email]")
    print("  weekly  (w)  - Show weekly analysis only")
    print("  monthly (m)  - Show monthly analysis only")
    print("  both         - Show both analyses (default)")
    print("  --no-email   - Skip sending email notification")
    print("")
    print("Email Configuration (Environment Variables):")
    print("  BOT_EMAIL_SENDER    - Gmail address for sending")
    print("  BOT_EMAIL_PASSWORD  - Gmail App Password")
    print("  BOT_EMAIL_RECIPIENT - Recipient email address")


def run_analysis(mode, symbol="BTCUSDT", balance=10000):
    """รันการวิเคราะห์และ return output"""

    all_output = ""

    if mode in ["weekly", "w", "both", "all"]:
        print("\n" + "━" * 100)
        print("                         📅 WEEKLY ANALYSIS")
        print("━" * 100 + "\n")

        weekly_trader = WeeklyTradingStrategy(symbol=symbol, leverage=5)

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        weekly_trader.get_weekly_recommendation(balance=balance)
        weekly_output = buffer.getvalue()
        sys.stdout = old_stdout

        print(weekly_output)
        all_output += "\n📅 WEEKLY ANALYSIS\n" + "=" * 50 + "\n" + weekly_output

    if mode in ["monthly", "m", "both", "all"]:
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
        all_output += "\n\n🌙 MONTHLY ANALYSIS\n" + "=" * 50 + "\n" + monthly_output

    return all_output


def send_email_notification(output, mode):
    """ส่งอีเมลแจ้งเตือน"""

    # Email configuration from environment variables
    email_sender = os.getenv("BOT_EMAIL_SENDER", "ong22280@gmail.com")
    email_password = os.getenv("BOT_EMAIL_PASSWORD", "tdkr anye gaam lped")
    email_recipient = os.getenv("BOT_EMAIL_RECIPIENT", "ong22280@gmail.com")

    if email_sender and email_password:
        notifier = EmailNotifier(email_sender, email_password, email_recipient)
        notifier.send_email(output, mode)
    else:
        print("\n" + "=" * 50)
        print("📧 EMAIL NOT CONFIGURED")
        print("   To enable email notifications:")
        print("   1. Set BOT_EMAIL_SENDER environment variable")
        print("   2. Set BOT_EMAIL_PASSWORD (Gmail App Password)")
        print("   3. Optionally set BOT_EMAIL_RECIPIENT")
        print("")
        print("   Example:")
        print("     export BOT_EMAIL_SENDER='ong22280@gmail.com'")
        print("     export BOT_EMAIL_PASSWORD='tdkr anye gaam lped'")
        print("=" * 50)


def main():
    """Main entry point"""

    # Configuration
    symbol = "BTCUSDT"
    balance = 10000

    # Parse arguments
    args = sys.argv[1:]

    # Determine mode
    mode = "both"
    for arg in args:
        if arg.lower() in ["weekly", "w", "monthly", "m", "both", "all"]:
            mode = arg.lower()
            break

    # Check for --no-email flag
    send_email = "--no-email" not in args

    # Check for help
    if "-h" in args or "--help" in args:
        print_usage()
        return

    # Print banner
    print_banner()

    # Validate mode
    if mode not in ["weekly", "w", "monthly", "m", "both", "all"]:
        print_usage()
        return

    # Run analysis
    output = run_analysis(mode, symbol, balance)

    # Send email if configured
    if send_email and output:
        send_email_notification(output, mode)


if __name__ == "__main__":
    main()

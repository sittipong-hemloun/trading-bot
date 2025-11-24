#!/bin/bash
# PythonAnywhere Setup Script
# Run this in PythonAnywhere Bash console

echo "🚀 Setting up Trading Bot on PythonAnywhere..."

# Clone repository (ถ้ายังไม่มี)
if [ ! -d "trading-bot" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/trading-bot.git
    cd trading-bot
else
    echo "📁 Repository exists, updating..."
    cd trading-bot
    git pull
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env file
echo "⚙️  Creating .env file..."
cat > .env << 'EOF'
BOT_EMAIL_SENDER=your@gmail.com
BOT_EMAIL_PASSWORD=your-app-password
BOT_EMAIL_RECIPIENT=recipient@email.com
DEEPSEEK_API_KEY=your-deepseek-key
EOF

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit .env file with your credentials:"
echo "   nano .env"
echo ""
echo "2. Test the bot:"
echo "   source venv/bin/activate"
echo "   python main.py deepseek --no-email"
echo ""
echo "3. Set up scheduled task in PythonAnywhere dashboard:"
echo "   - Go to 'Tasks' tab"
echo "   - Add new scheduled task"
echo "   - Command: cd ~/trading-bot && source venv/bin/activate && python main.py deepseek"
echo "   - Schedule: Daily at 23:30 UTC"
echo ""
echo "🎉 Done!"

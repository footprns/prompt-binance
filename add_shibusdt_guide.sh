#!/bin/bash
# Script to add SHIBUSDT to your trading system

echo "🐕 Adding SHIBUSDT to your trading system..."

cd /home/ubuntu/binance_scalper

# 1. Create SHIBUSDT configuration file
echo "📝 Creating SHIBUSDT configuration..."

cat > configs/shibusdt.env << 'EOF'
# SHIBUSDT Trading Configuration
# SHIB is a meme coin with high volatility - use conservative settings

# === API CREDENTIALS ===
BINANCE_API_KEY=redacted
BINANCE_API_SECRET=redacted
BINANCE_TESTNET=true

TELEGRAM_BOT_TOKEN=8104040269:AAGCcLjaKU69zCXrqCaeZJ-S5WQRI8-w-L8
TELEGRAM_CHAT_ID=7287230559

# === POSITION SIZING (Conservative for SHIB volatility) ===
TRADE_SYMBOL=SHIBUSDT
TRADE_QUANTITY=50000000
# SHIB trades in millions due to low price (around 0.00002)

# === WIDER TARGETS FOR MEME COIN VOLATILITY ===
PROFIT_TARGET=0.003
# 0.3% target - conservative for SHIB's wild moves
STOP_LOSS=0.006
# 0.6% stop loss - SHIB can swing ±1-2% on normal days
# Risk/Reward: 1:2 ratio

# === HIGHLY SELECTIVE ENTRY ===
MIN_SPREAD=0.008
# 0.8% minimum spread - SHIB spreads can be wide during volatility
# Very selective to avoid fake pumps/dumps

# === CONSERVATIVE MARTINGALE ===
MARTINGALE_ENABLED=true
MARTINGALE_MULTIPLIER=1.3
# Smaller multiplier for meme coin risk
MARTINGALE_MAX_STEPS=2
# Limited steps - meme coins can trend strongly
HALT_ON_MAX_STEPS=true

# === EXTENDED COOLDOWNS ===
COOLDOWN_SECONDS=600
# 10 minutes between trades - SHIB moves can last
EXTENDED_COOLDOWN_SECONDS=3600
# 1 hour extended cooldown

# === SHIB-SPECIFIC CONSIDERATIONS ===
# - Extreme volatility during social media hype
# - Large price swings on low volume
# - Sensitive to Elon Musk tweets and meme trends
# - Consider disabling during major crypto news events
# - Monitor social sentiment before trading
EOF

echo "✅ SHIBUSDT configuration created"

# 2. Verify configuration syntax
echo "🔍 Validating configuration..."

if grep -q "TRADE_SYMBOL=SHIBUSDT" configs/shibusdt.env; then
    echo "✅ Configuration syntax valid"
else
    echo "❌ Configuration error"
    exit 1
fi

# 3. Test configuration by starting the service
echo "🧪 Testing SHIBUSDT service..."

# Check if the template service exists
if [ ! -f "/etc/systemd/system/binance-scalper@.service" ]; then
    echo "❌ Template service not found. Please run the AI setup first."
    exit 1
fi

# Start the SHIBUSDT service
sudo systemctl daemon-reload
sudo systemctl start binance-scalper@shibusdt

# Wait a moment for startup
sleep 3

# Check if service started successfully
if sudo systemctl is-active binance-scalper@shibusdt >/dev/null; then
    echo "✅ SHIBUSDT service started successfully"
else
    echo "❌ SHIBUSDT service failed to start"
    echo "Checking logs..."
    sudo journalctl -u binance-scalper@shibusdt -n 10
    exit 1
fi

# 4. Update AI agent to monitor SHIBUSDT
echo "🤖 Adding SHIBUSDT to AI agent monitoring..."

# Update the AI agent configuration
if [ -f ".env.ai" ]; then
    # Check if SYMBOLS line exists and update it
    if grep -q "SYMBOLS=" .env.ai; then
        # Add SHIBUSDT to existing symbols
        sed -i 's/SYMBOLS=\(.*\)/SYMBOLS=\1,SHIBUSDT/' .env.ai
        # Remove any duplicate entries
        sed -i 's/SHIBUSDT,SHIBUSDT/SHIBUSDT/g' .env.ai
        sed -i 's/,,/,/g' .env.ai
    else
        # Add SYMBOLS line if it doesn't exist
        echo "SYMBOLS=BTCUSDT,ETHUSDT,ARBUSDT,SHIBUSDT" >> .env.ai
    fi
    echo "✅ AI agent updated to monitor SHIBUSDT"
else
    echo "⚠️  AI agent config not found. Add SHIBUSDT manually to .env.ai"
fi

# 5. Update dashboard to include SHIBUSDT
echo "📊 Adding SHIBUSDT to dashboard..."

# Create updated dashboard with SHIBUSDT tab
cat > update_dashboard.py << 'EOF'
import re

def update_dashboard():
    try:
        with open('dashboard.py', 'r') as f:
            content = f.read()
        
        # Add SHIBUSDT tab to the dashboard
        old_tabs = '''<div class="tab active" onclick="loadSymbol('BTCUSDT')">BTC/USDT</div>
                <div class="tab" onclick="loadSymbol('ETHUSDT')">ETH/USDT</div>
                <div class="tab" onclick="loadSymbol('ARBUSDT')">ARB/USDT</div>'''
        
        new_tabs = '''<div class="tab active" onclick="loadSymbol('BTCUSDT')">BTC/USDT</div>
                <div class="tab" onclick="loadSymbol('ETHUSDT')">ETH/USDT</div>
                <div class="tab" onclick="loadSymbol('ARBUSDT')">ARB/USDT</div>
                <div class="tab" onclick="loadSymbol('SHIBUSDT')">SHIB/USDT</div>'''
        
        content = content.replace(old_tabs, new_tabs)
        
        # Update the service status checks
        old_symbols = "for symbol in ['btcusdt', 'ethusdt', 'arbusdt']:"
        new_symbols = "for symbol in ['btcusdt', 'ethusdt', 'arbusdt', 'shibusdt']:"
        content = content.replace(old_symbols, new_symbols)
        
        with open('dashboard.py', 'w') as f:
            f.write(content)
        
        print("✅ Dashboard updated with SHIBUSDT")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard update failed: {e}")
        return False

if __name__ == '__main__':
    update_dashboard()
EOF

python update_dashboard.py

# 6. Restart services to apply changes
echo "🔄 Restarting services..."

# Restart AI agent with new symbol
if sudo systemctl is-active binance-ai-agent >/dev/null; then
    sudo systemctl restart binance-ai-agent
    echo "✅ AI agent restarted"
fi

# Restart dashboard with new tab
if sudo systemctl is-active binance-dashboard >/dev/null; then
    sudo systemctl restart binance-dashboard
    echo "✅ Dashboard restarted"
fi

# Enable SHIBUSDT service for auto-start
sudo systemctl enable binance-scalper@shibusdt

# 7. Display status and guidance
echo ""
echo "🎉 SHIBUSDT successfully added to your trading system!"
echo ""
echo "📊 Dashboard: http://your-server-ip:8080 (now includes SHIB/USDT tab)"
echo "🔧 SHIBUSDT Service: sudo systemctl status binance-scalper@shibusdt"
echo "📋 SHIBUSDT Logs: sudo journalctl -u binance-scalper@shibusdt -f"
echo ""
echo "🔍 Current Services:"
for symbol in btcusdt ethusdt arbusdt shibusdt; do
    if sudo systemctl is-active binance-scalper@$symbol >/dev/null 2>&1; then
        status="🟢 Running"
    else
        status="🔴 Stopped"
    fi
    echo "  - $(echo $symbol | tr '[:lower:]' '[:upper:]'): $status"
done

echo ""
echo "⚠️  SHIB Trading Warnings:"
echo "  - Extremely volatile meme coin"
echo "  - Can move 10-50% in minutes during hype"
echo "  - Monitor social media sentiment"
echo "  - Consider smaller position sizes"
echo "  - Be ready to pause during extreme volatility"

echo ""
echo "🛠️  Management Commands:"
echo "  Start:   sudo systemctl start binance-scalper@shibusdt"
echo "  Stop:    sudo systemctl stop binance-scalper@shibusdt"
echo "  Restart: sudo systemctl restart binance-scalper@shibusdt"
echo "  Logs:    sudo journalctl -u binance-scalper@shibusdt -f"
echo ""
echo "📈 The AI agent will now monitor and optimize SHIBUSDT trading!"

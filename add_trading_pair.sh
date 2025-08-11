#!/bin/bash
# Universal Trading Pair Addition Tool
# Enhanced version with security validation and flexible configuration
# Author: AI Trading System
# Version: 2.0

set -euo pipefail  # Enhanced error handling for production environments

# Color codes for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="/home/ubuntu/binance_scalper"
CONFIG_DIR="$SCRIPT_DIR/configs"
BACKUP_DIR="$CONFIG_DIR/backups"
LOG_FILE="$SCRIPT_DIR/logs/pair_addition.log"

# Security and validation settings
MAX_QUANTITY_THRESHOLD=1000000000  # 1B max quantity for safety
MIN_SPREAD_THRESHOLD=0.001         # 0.1% minimum spread
MAX_SPREAD_THRESHOLD=0.02          # 2% maximum spread

# Logging function with timestamps
log() {
    local level=$1
    shift
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*" | tee -a "$LOG_FILE"
}

# Security validation function
validate_symbol() {
    local symbol=$1
    
    # Check symbol format (should be like BTCUSDT, ETHUSDT, etc.)
    if [[ ! $symbol =~ ^[A-Z0-9]{3,10}USDT$ ]]; then
        log "ERROR" "Invalid symbol format: $symbol. Must be like BTCUSDT, ETHUSDT, etc."
        return 1
    fi
    
    # Check for potentially dangerous symbols (basic security check)
    local dangerous_patterns=("SCAM" "FAKE" "TEST" "HACK")
    for pattern in "${dangerous_patterns[@]}"; do
        if [[ $symbol == *"$pattern"* ]]; then
            log "WARN" "Symbol contains potentially risky pattern: $pattern"
            read -p "Are you sure you want to continue? (yes/no): " confirm
            if [[ $confirm != "yes" ]]; then
                return 1
            fi
        fi
    done
    
    return 0
}

# Function to get trading pair characteristics from user or API
get_pair_config() {
    local symbol_upper=$1
    local symbol_lower=$2
    local config_file="$CONFIG_DIR/${symbol_lower}.env"
    
    echo -e "${CYAN}📊 Configuring trading parameters for $symbol_upper${NC}"
    echo
    
    # Determine pair type for default settings
    local pair_type="standard"
    case $symbol_upper in
        *SHIB*|*DOGE*|*PEPE*|*FLOKI*)
            pair_type="meme"
            echo -e "${YELLOW}⚠️  Detected meme coin - using conservative settings${NC}"
            ;;
        *BTC*|*ETH*|*BNB*)
            pair_type="major"
            echo -e "${GREEN}✅ Detected major cryptocurrency${NC}"
            ;;
        *ADA*|*DOT*|*LINK*|*UNI*)
            pair_type="altcoin"
            echo -e "${BLUE}📈 Detected established altcoin${NC}"
            ;;
    esac
    
    # Get quantity (most important parameter)
    echo -e "${PURPLE}💰 Trade Quantity Configuration${NC}"
    echo "This determines your position size per trade."
    
    local default_quantity
    case $pair_type in
        "meme")
            case $symbol_upper in
                *SHIB*) default_quantity=50000000 ;;
                *DOGE*) default_quantity=1000 ;;
                *PEPE*) default_quantity=100000000 ;;
                *) default_quantity=10000000 ;;
            esac
            ;;
        "major")
            default_quantity=100
            ;;
        "altcoin")
            default_quantity=200
            ;;
        *)
            default_quantity=100
            ;;
    esac
    
    read -p "Enter trade quantity (default: $default_quantity): " quantity
    quantity=${quantity:-$default_quantity}
    
    # Validate quantity
    if ! [[ "$quantity" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$quantity > $MAX_QUANTITY_THRESHOLD" | bc -l) )); then
        log "ERROR" "Invalid quantity: $quantity (max: $MAX_QUANTITY_THRESHOLD)"
        return 1
    fi
    
    # Get risk parameters
    echo -e "${PURPLE}🎯 Risk Management Parameters${NC}"
    
    # Set default values based on pair type
    local default_profit_target default_stop_loss default_min_spread
    case $pair_type in
        "meme")
            default_profit_target=0.004   # 0.4% - meme coins are volatile
            default_stop_loss=0.008       # 0.8% - wider stops
            default_min_spread=0.008      # 0.8% - require good spread
            ;;
        "major")
            default_profit_target=0.0025  # 0.25% - tight for majors
            default_stop_loss=0.006       # 0.6% - moderate stops
            default_min_spread=0.004      # 0.4% - standard spread
            ;;
        "altcoin")
            default_profit_target=0.003   # 0.3% - moderate target
            default_stop_loss=0.007       # 0.7% - moderate stops
            default_min_spread=0.006      # 0.6% - selective entry
            ;;
        *)
            default_profit_target=0.003   # 0.3% - fallback default
            default_stop_loss=0.007       # 0.7% - fallback default
            default_min_spread=0.006      # 0.6% - fallback default
            ;;
    esac
    
    # Calculate percentage displays
    local profit_pct_display=$(echo "scale=2; $default_profit_target * 100" | bc -l)
    local stop_pct_display=$(echo "scale=2; $default_stop_loss * 100" | bc -l)
    local spread_pct_display=$(echo "scale=2; $default_min_spread * 100" | bc -l)
    
    read -p "Profit target % (default: ${profit_pct_display}%): " profit_input
    profit_target=${profit_input:-$profit_pct_display}
    profit_target=$(echo "scale=6; $profit_target / 100" | bc -l)
    
    read -p "Stop loss % (default: ${stop_pct_display}%): " stop_input
    stop_loss=${stop_input:-$stop_pct_display}
    stop_loss=$(echo "scale=6; $stop_loss / 100" | bc -l)
    
    read -p "Minimum spread % (default: ${spread_pct_display}%): " spread_input
    min_spread=${spread_input:-$spread_pct_display}
    min_spread=$(echo "scale=6; $min_spread / 100" | bc -l)
    
    # Validate risk parameters
    if (( $(echo "$min_spread < $MIN_SPREAD_THRESHOLD || $min_spread > $MAX_SPREAD_THRESHOLD" | bc -l) )); then
        log "ERROR" "Invalid spread: $min_spread (range: $MIN_SPREAD_THRESHOLD - $MAX_SPREAD_THRESHOLD)"
        return 1
    fi
    
    # Timing parameters
    echo -e "${PURPLE}⏰ Timing Configuration${NC}"
    
    # Set default timing values based on pair type
    local default_cooldown default_extended_cooldown
    case $pair_type in
        "meme")
            default_cooldown=600      # 10 minutes - meme coins need cooling
            default_extended_cooldown=3600  # 1 hour
            ;;
        "major")
            default_cooldown=300      # 5 minutes
            default_extended_cooldown=1800  # 30 minutes
            ;;
        "altcoin")
            default_cooldown=450      # 7.5 minutes
            default_extended_cooldown=2700  # 45 minutes
            ;;
        *)
            default_cooldown=400      # 6.67 minutes - fallback default
            default_extended_cooldown=2400  # 40 minutes - fallback default
            ;;
    esac
    
    read -p "Cooldown seconds (default: $default_cooldown): " cooldown
    cooldown=${cooldown:-$default_cooldown}
    
    read -p "Extended cooldown seconds (default: $default_extended_cooldown): " extended_cooldown
    extended_cooldown=${extended_cooldown:-$default_extended_cooldown}
    
    # Martingale parameters
    echo -e "${PURPLE}📈 Martingale Configuration${NC}"
    echo "Martingale can amplify both profits and losses. Be conservative."
    
    read -p "Enable martingale? (true/false, default: true): " martingale_enabled
    martingale_enabled=${martingale_enabled:-true}
    
    if [[ $martingale_enabled == "true" ]]; then
        local default_multiplier=1.5
        local default_max_steps=2
        
        # More conservative for meme coins
        if [[ $pair_type == "meme" ]]; then
            default_multiplier=1.3
            default_max_steps=2
        fi
        
        read -p "Martingale multiplier (default: $default_multiplier): " martingale_multiplier
        martingale_multiplier=${martingale_multiplier:-$default_multiplier}
        
        read -p "Max martingale steps (default: $default_max_steps): " martingale_max_steps
        martingale_max_steps=${martingale_max_steps:-$default_max_steps}
    else
        martingale_multiplier=1.0
        martingale_max_steps=0
    fi
    
    # Create configuration file with security headers
    log "INFO" "Creating configuration file for $symbol_upper"
    
    cat > "$config_file" << EOF
# $symbol_upper Trading Configuration
# Created: $(date)
# Pair Type: $pair_type
# SECURITY: This file contains trading parameters - handle with care

# === API CREDENTIALS ===
BINANCE_API_KEY=redacted
BINANCE_API_SECRET=redacted
BINANCE_TESTNET=true

TELEGRAM_BOT_TOKEN=redacted
TELEGRAM_CHAT_ID=redacted

# === POSITION SIZING ===
TRADE_SYMBOL=$symbol_upper
TRADE_QUANTITY=$quantity

# === RISK MANAGEMENT ===
PROFIT_TARGET=$profit_target
STOP_LOSS=$stop_loss
# Risk/Reward Ratio: $(echo "scale=2; $profit_target / $stop_loss" | bc):1

# === ENTRY REQUIREMENTS ===
MIN_SPREAD=$min_spread
# Minimum spread required for trade entry

# === MARTINGALE SYSTEM ===
MARTINGALE_ENABLED=$martingale_enabled
MARTINGALE_MULTIPLIER=$martingale_multiplier
MARTINGALE_MAX_STEPS=$martingale_max_steps
HALT_ON_MAX_STEPS=true

# === TIMING CONTROLS ===
COOLDOWN_SECONDS=$cooldown
EXTENDED_COOLDOWN_SECONDS=$extended_cooldown

# === $symbol_upper SPECIFIC NOTES ===
EOF

    # Add pair-specific warnings
    case $pair_type in
        "meme")
            cat >> "$config_file" << EOF
# ⚠️ MEME COIN WARNINGS:
# - Extremely volatile and sentiment-driven
# - Can move 50%+ in minutes during hype cycles
# - Monitor social media and news closely
# - Consider pausing during major market events
# - Position sizing is critical - start small
EOF
            ;;
        "major")
            cat >> "$config_file" << EOF
# 💎 MAJOR CRYPTOCURRENCY:
# - Generally more stable than altcoins
# - Higher liquidity and tighter spreads
# - Still subject to market-wide volatility
# - Good for consistent scalping strategies
EOF
            ;;
        "altcoin")
            cat >> "$config_file" << EOF
# 📊 ALTCOIN TRADING:
# - Moderate volatility and liquidity
# - May be influenced by Bitcoin movements
# - Check for upcoming events and announcements
# - Adjust position size based on market cap
EOF
            ;;
    esac
    
    echo -e "${GREEN}✅ Configuration created successfully${NC}"
    return 0
}

# Function to update dashboard
update_dashboard() {
    local symbol_upper=$1
    local symbol_lower=$2
    local dashboard_file="$SCRIPT_DIR/dashboard.py"
    
    if [[ ! -f $dashboard_file ]]; then
        log "WARN" "Dashboard file not found: $dashboard_file"
        return 1
    fi
    
    log "INFO" "Updating dashboard to include $symbol_upper"
    
    # Create backup
    cp "$dashboard_file" "$dashboard_file.backup.$(date +%s)"
    
    # Check if symbol already exists in dashboard
    if grep -q "onclick=\"loadSymbol('$symbol_upper')" "$dashboard_file"; then
        log "INFO" "$symbol_upper already exists in dashboard"
        return 0
    fi
    
    # Add symbol to dashboard tabs (find the last tab and add after it)
    local last_tab_line=$(grep -n "onclick=\"loadSymbol(" "$dashboard_file" | tail -1 | cut -d: -f1)
    
    if [[ -n $last_tab_line ]]; then
        # Insert new tab after the last existing tab
        local symbol_display="${symbol_upper%USDT}"
        sed -i "${last_tab_line}a\\                <div class=\"tab\" onclick=\"loadSymbol('$symbol_upper')\">${symbol_display}/USDT</div>" "$dashboard_file"
        
        # Also add to service status checks (use lowercase for service names)
        # Find the line with the symbol list and add the new symbol properly
        if grep -q "for symbol in \[.*'bnbusdt'\]:" "$dashboard_file"; then
            sed -i "s/'bnbusdt'\]/'bnbusdt', '${symbol_lower}']/" "$dashboard_file"
            log "INFO" "Added $symbol_lower to dashboard service status checks"
        else
            log "WARN" "Could not find service status check list to update"
        fi
        
        log "INFO" "Dashboard updated with $symbol_upper tab"
    else
        log "WARN" "Could not automatically update dashboard - manual update required"
    fi
}

# Function to update AI agent
update_ai_agent() {
    local symbol_upper=$1
    local ai_config="$SCRIPT_DIR/.env.ai"
    
    log "INFO" "Adding $symbol_upper to AI agent monitoring"
    
    if [[ -f $ai_config ]]; then
        # Backup AI config
        cp "$ai_config" "$ai_config.backup.$(date +%s)"
        
        # Check if SYMBOLS line exists
        if grep -q "^SYMBOLS=" "$ai_config"; then
            # Add symbol to existing list (if not already present)
            if ! grep -q "$symbol_upper" "$ai_config"; then
                sed -i "s/SYMBOLS=\(.*\)/SYMBOLS=\1,$symbol_upper/" "$ai_config"
                # Clean up any double commas
                sed -i 's/,,/,/g' "$ai_config"
                log "INFO" "Added $symbol_upper to AI agent symbol list"
            else
                log "INFO" "$symbol_upper already in AI agent monitoring"
            fi
        else
            # Add SYMBOLS line
            echo "SYMBOLS=$symbol_upper" >> "$ai_config"
            log "INFO" "Created AI agent symbol list with $symbol_upper"
        fi
    else
        log "WARN" "AI agent config not found: $ai_config"
        return 1
    fi
}

# Function to test service
test_service() {
    local symbol_upper=$1
    local symbol_lower=$2
    local service_name="binance-scalper@${symbol_lower}"
    
    log "INFO" "Testing $service_name service"
    
    # Check if template service exists
    if [[ ! -f "/etc/systemd/system/binance-scalper@.service" ]]; then
        log "ERROR" "Template service not found. Run main setup first."
        return 1
    fi
    
    # Reload systemd and start service
    sudo systemctl daemon-reload
    sudo systemctl start "$service_name"
    
    # Wait for startup
    sleep 5
    
    # Check service status
    if sudo systemctl is-active "$service_name" >/dev/null; then
        log "SUCCESS" "$service_name started successfully"
        
        # Enable for auto-start
        sudo systemctl enable "$service_name"
        log "INFO" "$service_name enabled for auto-start"
        
        return 0
    else
        log "ERROR" "$service_name failed to start"
        echo -e "${RED}Service logs:${NC}"
        sudo journalctl -u "$service_name" -n 20 --no-pager
        return 1
    fi
}

# Function to show service management commands
show_management_commands() {
    local symbol_upper=$1
    local symbol_lower=$2
    local service_name="binance-scalper@${symbol_lower}"
    
    echo -e "${CYAN}🛠️  Service Management Commands for $symbol_upper:${NC}"
    echo -e "${GREEN}Start:   ${NC}sudo systemctl start $service_name"
    echo -e "${RED}Stop:    ${NC}sudo systemctl stop $service_name"
    echo -e "${YELLOW}Restart: ${NC}sudo systemctl restart $service_name"
    echo -e "${BLUE}Status:  ${NC}sudo systemctl status $service_name"
    echo -e "${PURPLE}Logs:    ${NC}sudo journalctl -u $service_name -f"
    echo -e "${CYAN}Enable:  ${NC}sudo systemctl enable $service_name"
    echo -e "${CYAN}Disable: ${NC}sudo systemctl disable $service_name"
    echo
}

# Function to show current system status
show_system_status() {
    echo -e "${CYAN}📊 Current Trading System Status:${NC}"
    echo
    
    # Check AI agent
    if sudo systemctl is-active binance-ai-agent >/dev/null 2>&1; then
        echo -e "🤖 AI Agent: ${GREEN}Running${NC}"
    else
        echo -e "🤖 AI Agent: ${RED}Stopped${NC}"
    fi
    
    # Check dashboard
    if sudo systemctl is-active binance-dashboard >/dev/null 2>&1; then
        echo -e "📊 Dashboard: ${GREEN}Running${NC}"
    else
        echo -e "📊 Dashboard: ${RED}Stopped${NC}"
    fi
    
    # Check trading services
    echo -e "\n📈 Trading Services:"
    for config_file in "$CONFIG_DIR"/*.env; do
        if [[ -f $config_file && $(basename "$config_file") != ".env_template" ]]; then
            local symbol=$(basename "$config_file" .env)
            local service_name="binance-scalper@$symbol"
            
            if sudo systemctl is-active "$service_name" >/dev/null 2>&1; then
                echo -e "   ${symbol^^}: ${GREEN}Running${NC}"
            else
                echo -e "   ${symbol^^}: ${RED}Stopped${NC}"
            fi
        fi
    done
    echo
}

# Main function
main() {
    # Check if running as correct user (security check)
    if [[ $EUID -eq 0 ]]; then
        log "ERROR" "Do not run this script as root for security reasons"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p "$CONFIG_DIR" "$BACKUP_DIR" "$(dirname "$LOG_FILE")"
    
    # Check if we're in the right directory
    if [[ ! -d "$CONFIG_DIR" ]]; then
        log "ERROR" "Config directory not found: $CONFIG_DIR"
        log "ERROR" "Please run this script from the binance_scalper directory"
        exit 1
    fi
    
    echo -e "${BLUE}🚀 Universal Trading Pair Addition Tool v2.0${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo
    
    # Show current status first
    show_system_status
    
    # Get symbol from user
    read -p "Enter trading pair symbol (e.g., ADAUSDT, LINKUSDT): " symbol_input
    
    # Normalize case - store both uppercase and lowercase versions
    SYMBOL_UPPER=$(echo "$symbol_input" | tr '[:lower:]' '[:upper:]')
    SYMBOL_LOWER=$(echo "$SYMBOL_UPPER" | tr '[:upper:]' '[:lower:]')
    
    # Use uppercase for display and validation
    symbol="$SYMBOL_UPPER"
    
    # Validate symbol
    if ! validate_symbol "$symbol"; then
        log "ERROR" "Symbol validation failed"
        exit 1
    fi
    
    # Check if symbol already exists (use lowercase for filename)
    local config_file="$CONFIG_DIR/${SYMBOL_LOWER}.env"
    if [[ -f $config_file ]]; then
        echo -e "${YELLOW}⚠️  Configuration for $symbol already exists${NC}"
        read -p "Do you want to overwrite it? (yes/no): " overwrite
        if [[ $overwrite != "yes" ]]; then
            log "INFO" "Operation cancelled by user"
            exit 0
        fi
        
        # Backup existing config
        cp "$config_file" "$BACKUP_DIR/${SYMBOL_LOWER}.env.backup.$(date +%s)"
        log "INFO" "Existing config backed up"
    fi
    
    echo -e "${CYAN}📝 Configuring $symbol for trading...${NC}"
    echo
    
    # Get configuration from user (pass both versions)
    if ! get_pair_config "$SYMBOL_UPPER" "$SYMBOL_LOWER"; then
        log "ERROR" "Configuration setup failed"
        exit 1
    fi
    
    # Test the configuration
    echo -e "${CYAN}🧪 Testing service configuration...${NC}"
    if ! test_service "$SYMBOL_UPPER" "$SYMBOL_LOWER"; then
        log "ERROR" "Service test failed"
        read -p "Continue anyway? (yes/no): " continue_anyway
        if [[ $continue_anyway != "yes" ]]; then
            exit 1
        fi
    fi
    
    # Update dashboard
    echo -e "${CYAN}📊 Updating dashboard...${NC}"
    update_dashboard "$SYMBOL_UPPER" "$SYMBOL_LOWER"
    
    # Update AI agent
    echo -e "${CYAN}🤖 Updating AI agent...${NC}"
    update_ai_agent "$SYMBOL_UPPER"
    
    # Restart services
    echo -e "${CYAN}🔄 Restarting related services...${NC}"
    
    # Restart AI agent if it was running
    if sudo systemctl is-active binance-ai-agent >/dev/null 2>&1; then
        sudo systemctl restart binance-ai-agent
        log "INFO" "AI agent restarted"
    fi
    
    # Restart dashboard if it was running
    if sudo systemctl is-active binance-dashboard >/dev/null 2>&1; then
        sudo systemctl restart binance-dashboard
        log "INFO" "Dashboard restarted"
    fi
    
    # Final success message
    echo
    echo -e "${GREEN}🎉 $SYMBOL_UPPER successfully added to your trading system!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo
    
    # Show management commands
    show_management_commands "$SYMBOL_UPPER" "$SYMBOL_LOWER"
    
    # Show access information
    echo -e "${CYAN}📊 Access Information:${NC}"
    echo -e "${BLUE}Dashboard: ${NC}http://$(hostname -I | cut -d' ' -f1):8080"
    echo -e "${BLUE}New Tab:   ${NC}${SYMBOL_UPPER%USDT}/USDT"
    echo
    
    # Show final status
    show_system_status
    
    log "SUCCESS" "$SYMBOL_UPPER trading pair addition completed"
}

# Run main function
main "$@"
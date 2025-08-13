import os
import time
import requests
import logging
import json
import websocket
import threading
import numpy as np
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from binance.client import Client
from trade_logger_integration import TradeLogger
from datetime import datetime, timedelta, timezone
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# === Configuration ===
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
QTY = float(os.getenv("TRADE_QUANTITY", "0.001"))
MIN_SPREAD = float(os.getenv("MIN_SPREAD", "0.006"))  # Increased for selectivity
PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", "0.003"))  # Improved R:R
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.005"))  # Tighter stop

# Enhanced parameters for win rate optimization
MIN_BOOK_DEPTH = float(os.getenv("MIN_BOOK_DEPTH", "1000"))  # Minimum order book depth
MAX_SPREAD_VOLATILITY = float(os.getenv("MAX_SPREAD_VOLATILITY", "0.0005"))  # Spread stability
HIGH_VOL_THRESHOLD = float(os.getenv("HIGH_VOL_THRESHOLD", "0.01"))  # 1% volatility threshold
LOW_VOL_THRESHOLD = float(os.getenv("LOW_VOL_THRESHOLD", "0.003"))  # 0.3% volatility threshold
ENABLE_TRAILING_STOP = os.getenv("ENABLE_TRAILING_STOP", "true").lower() == "true"
ENABLE_TIME_FILTER = os.getenv("ENABLE_TIME_FILTER", "true").lower() == "true"
ENABLE_MARKET_REGIME = os.getenv("ENABLE_MARKET_REGIME", "true").lower() == "true"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Cooldown & Martingale
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "600"))  # Increased to 10 minutes

MARTINGALE_ENABLED = os.getenv("MARTINGALE_ENABLED", "false").lower() == "true"  # Disabled by default
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "1.5"))
MARTINGALE_MAX_STEPS = int(os.getenv("MARTINGALE_MAX_STEPS", "2"))  # Reduced

HALT_ON_MAX_STEPS = os.getenv("HALT_ON_MAX_STEPS", "true").lower() == "true"
EXTENDED_COOLDOWN_SECONDS = int(os.getenv("EXTENDED_COOLDOWN_SECONDS", "1800"))  # 30 minutes

# Fee-aware enhancements for the scalper

# Add these constants at the top of your scalper file
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.001"))      # 0.1% default
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.001"))      # 0.1% default  
BNB_FEE_DISCOUNT = float(os.getenv("BNB_FEE_DISCOUNT", "0.25"))  # 25% discount with BNB
PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", "0.002"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.001"))
# Fee configuration
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.001"))      # 0.1%
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.001"))      # 0.1%
BNB_DISCOUNT = float(os.getenv("BNB_DISCOUNT", "0.25"))  # 25%
ROUND_TRIP_FEE = (MAKER_FEE + TAKER_FEE) * (1 - BNB_DISCOUNT)  # Estimated fee


class EnhancedBinanceScalper:
    def __init__(self):
        # Initialize Binance client
        self.client = Client(API_KEY, API_SECRET, testnet=TESTNET)

        # WebSocket connections
        self.ws = None
        self.depth_ws = None
        self.trade_ws = None

        # Trade state
        self.trade_active = False
        self.trading_paused = False
        self.entry_price = 0.0
        self.target_price = 0.0
        self.stop_price = 0.0
        self.current_price = 0.0
        self.position_qty = 0.0

        # Martingale state
        self.base_qty = QTY
        self.current_qty = QTY
        self.martingale_step = 0

        # Cooldown state
        self.cooldown_until = 0.0

        # Symbol info for precision
        self.symbol_info = None
        self.price_precision = 8
        self.quantity_precision = 8
        self.base_asset = "BASE"
        self.quote_asset = "QUOTE"
        self.min_qty = None
        self.max_qty = None
        self.market_min_qty = None
        self.market_max_qty = None
        self.min_notional = None
        self.max_num_orders = None

        # Enhanced win rate optimization features
        self.spread_history = deque(maxlen=200)  # Track spread history
        self.price_history = deque(maxlen=100)   # Track price history for volatility
        self.trade_count_by_hour = {}  # Track performance by hour
        self.last_depth_data = None
        self.market_regime = "NORMAL"
        self.confidence_score = 0.0
        
        # Trailing stop state
        self.original_stop_price = 0.0
        self.highest_price_since_entry = 0.0

        # Initialize symbol info
        self._get_symbol_info()

        self.trade_logger = TradeLogger()
        self.spread_at_entry = 0.0

    # ---------- Enhanced Market Analysis ----------
    
    def calculate_recent_volatility(self, periods=20):
        """Calculate recent price volatility"""
        if len(self.price_history) < periods:
            return 0.0
        
        prices = list(self.price_history)[-periods:]
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return np.std(returns) if returns else 0.0
    
    def detect_market_regime(self):
        """Detect current market volatility regime"""
        if not ENABLE_MARKET_REGIME:
            return "NORMAL"
        
        volatility = self.calculate_recent_volatility()
        
        if volatility > HIGH_VOL_THRESHOLD:
            return "HIGH_VOL"
        elif volatility < LOW_VOL_THRESHOLD:
            return "LOW_VOL"
        else:
            return "NORMAL"
    
    def is_good_trading_time(self):
        """Check if current time is favorable for trading"""
        if not ENABLE_TIME_FILTER:
            return True
        
        current_hour = datetime.now(timezone.utc).hour
        
        # Avoid low liquidity periods (Asian early morning UTC)
        if 2 <= current_hour <= 6:
            logger.debug(f"Avoiding low liquidity period: {current_hour}:00 UTC")
            return False
        
        # Prefer high activity periods (European/US overlap)
        if 12 <= current_hour <= 18:
            return True
        
        # Also good during US session
        if 19 <= current_hour <= 23:
            return True
        
        # Moderate activity other times
        return True
    
    def analyze_order_book_depth(self, bids, asks):
        """Analyze order book depth for entry quality"""
        if not bids or not asks or len(bids) < 5 or len(asks) < 5:
            return False, 0.0
        
        # Calculate depth in quote asset (USDT)
        bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids[:5])
        ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks[:5])
        
        total_depth = bid_depth + ask_depth
        
        # Check minimum depth requirement
        if total_depth < MIN_BOOK_DEPTH:
            logger.debug(f"Insufficient book depth: {total_depth:.2f} < {MIN_BOOK_DEPTH}")
            return False, 0.0
        
        # Calculate depth balance (prefer balanced books)
        depth_ratio = min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
        
        return True, depth_ratio
    
    def analyze_spread_quality(self, bids, asks):
            """Enhanced spread analysis with debugging"""
            if not bids or not asks:
                logger.debug("No bids or asks available")
                return False, 0.0, 0.0
            
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                # DEBUG: Log raw values
                logger.debug(f"Raw bid/ask: {best_bid}/{best_ask}")
                
                # Sanity check: ask should be higher than bid
                if best_ask <= best_bid:
                    logger.warning(f"Invalid market data: ask {best_ask} <= bid {best_bid}")
                    return False, 0.0, 0.0
                
                spread = (best_ask - best_bid) / best_bid
                
                # DEBUG: Log calculated spread
                logger.debug(f"Calculated spread: {spread:.6f} ({spread*100:.4f}%)")
                
                # Add to spread history only if positive
                if spread > 0:
                    self.spread_history.append(spread)
                else:
                    logger.warning(f"Negative spread calculated: {spread:.6f}")
                    return False, spread, 0.0
                
                # Check basic spread requirement
                if spread < MIN_SPREAD:
                    logger.debug(f"Spread too small: {spread:.6f} < {MIN_SPREAD}")
                    return False, spread, 0.0
                
                # Check spread stability (avoid volatile spreads)
                if len(self.spread_history) >= 10:
                    recent_spreads = list(self.spread_history)[-10:]
                    spread_volatility = np.std(recent_spreads)
                    
                    if spread_volatility > MAX_SPREAD_VOLATILITY:
                        logger.debug(f"Spread too volatile: {spread_volatility:.6f} > {MAX_SPREAD_VOLATILITY}")
                        return False, spread, spread_volatility
                
                # Check if spread is abnormally wide (potential news event)
                if len(self.spread_history) >= 50:
                    avg_spread = np.mean(list(self.spread_history)[-50:])
                    if spread > avg_spread * 2.5:
                        logger.debug(f"Spread abnormally wide: {spread:.6f} > {avg_spread * 2.5:.6f}")
                        return False, spread, 0.0
                
                logger.debug(f"Spread quality check passed: {spread:.6f}")
                return True, spread, np.std(list(self.spread_history)[-10:]) if len(self.spread_history) >= 10 else 0.0
                
            except Exception as e:
                logger.error(f"Error in spread calculation: {e}")
                return False, 0.0, 0.0

    def calculate_confidence_score(self, spread, depth_ratio, market_regime):
        """Calculate confidence score for trade entry"""
        score = 0.5  # Base score
        
        # Spread quality (0-0.3 points)
        if len(self.spread_history) >= 20:
            avg_spread = np.mean(list(self.spread_history)[-20:])
            spread_percentile = spread / avg_spread
            if spread_percentile > 1.2:  # Above average spread
                score += 0.2
            elif spread_percentile > 1.5:  # Well above average
                score += 0.3
        
        # Book depth quality (0-0.2 points)
        if depth_ratio > 0.8:  # Well balanced book
            score += 0.2
        elif depth_ratio > 0.6:
            score += 0.1
        
        # Market regime (0-0.2 points)
        if market_regime == "LOW_VOL":
            score += 0.2  # Low volatility is good for scalping
        elif market_regime == "NORMAL":
            score += 0.1
        # HIGH_VOL gets no bonus (riskier)
        
        # Time of day bonus (0-0.1 points)
        current_hour = datetime.now(timezone.utc).hour
        if 12 <= current_hour <= 18 or 19 <= current_hour <= 23:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def calculate_position_size(self, confidence_score):
        """Calculate position size based on confidence and market regime"""
        base_size = self.current_qty
        
        # Adjust for confidence
        if confidence_score > 0.8:
            size_multiplier = 1.1  # Slightly larger for high confidence
        elif confidence_score < 0.6:
            size_multiplier = 0.8  # Smaller for low confidence
        else:
            size_multiplier = 1.0
        
        # Adjust for market regime
        if self.market_regime == "HIGH_VOL":
            size_multiplier *= 0.7  # Smaller size in volatile markets
        elif self.market_regime == "LOW_VOL":
            size_multiplier *= 1.1  # Slightly larger in calm markets
        
        return base_size * size_multiplier
    
    def update_trailing_stop(self):
        """Update trailing stop loss"""
        if not ENABLE_TRAILING_STOP or not self.trade_active:
            return
        
        # Track highest price since entry
        if self.current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = self.current_price
        
        # Calculate current profit
        current_profit_pct = (self.current_price - self.entry_price) / self.entry_price
        
        if current_profit_pct > 0.001:  # 0.1% profit
            # Trail stop to breakeven plus small buffer
            breakeven_stop = self.entry_price * 1.0002  # 0.02% above entry
            self.stop_price = max(self.stop_price, breakeven_stop)
            
        if current_profit_pct > 0.002:  # 0.2% profit
            # Trail stop to secure some profit
            profit_stop = self.entry_price * 1.001  # 0.1% profit secured
            self.stop_price = max(self.stop_price, profit_stop)

    # ---------- Enhanced Entry Logic ----------
    
    def handle_depth_data(self, data):
        """Enhanced depth stream handler with comprehensive debugging"""
        if self.trade_active or self.in_cooldown() or self.trading_paused:
            if self.trade_active:
                logger.debug("Skipping entry - trade already active")
            elif self.in_cooldown():
                remaining = int(self.cooldown_until - self.now())
                logger.debug(f"Skipping entry - cooldown for {remaining}s")
            else:
                logger.debug("Skipping entry - trading paused")
            return

        try:
            bids = data.get("b", [])
            asks = data.get("a", [])
            
            if not bids or not asks:
                logger.debug("Empty bids or asks in depth data")
                return

            # Parse and validate order book data with enhanced debugging
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                # Check for inverted data and attempt to fix
                if best_ask <= best_bid:
                    logger.warning(f"Inverted order book detected: ask {best_ask} <= bid {best_bid}")
                    
                    # Try swapping bid/ask arrays first
                    if len(asks) > 0 and len(bids) > 0:
                        try:
                            alt_best_bid = float(asks[0][0])
                            alt_best_ask = float(bids[0][0])
                            
                            if alt_best_ask > alt_best_bid:
                                logger.info(f"Fixed order book by swapping: bid={alt_best_bid}, ask={alt_best_ask}")
                                bids, asks = asks, bids
                                best_bid, best_ask = alt_best_bid, alt_best_ask
                            else:
                                logger.warning("Could not fix order book, skipping this update")
                                return
                        except Exception as e:
                            logger.warning(f"Failed to fix order book: {e}")
                            return
                    else:
                        logger.warning("Insufficient data to fix order book")
                        return
                
            except (IndexError, ValueError, TypeError) as e:
                logger.error(f"Error parsing order book data: {e}")
                return

            # Update current price
            mid_price = (best_bid + best_ask) / 2.0
            self.current_price = mid_price
            self.price_history.append(mid_price)
            
            # Update market regime
            self.market_regime = self.detect_market_regime()
            
            # Enhanced debugging for each filter
            logger.debug(f"Market check - Price: {mid_price:.4f}, Regime: {self.market_regime}")
            
            # Time filter check
            if not self.is_good_trading_time():
                current_hour = datetime.now(timezone.utc).hour
                logger.debug(f"Time filter failed - Current hour: {current_hour}")
                return
            
            # Order book depth analysis
            depth_ok, depth_ratio = self.analyze_order_book_depth(bids, asks)
            if not depth_ok:
                logger.debug(f"Depth check failed - Ratio: {depth_ratio:.2f}")
                return
            
            # Enhanced spread analysis
            spread_ok, spread, spread_volatility = self.analyze_spread_quality(bids, asks)
            if not spread_ok:
                logger.debug(f"Spread check failed - Spread: {spread:.6f} ({spread*100:.3f}%), Required: {MIN_SPREAD*100:.3f}%")
                return
            
            # Calculate confidence score
            self.confidence_score = self.calculate_confidence_score(spread, depth_ratio, self.market_regime)
            
            # Require minimum confidence for entry
            min_confidence = 0.6
            if self.market_regime == "HIGH_VOL":
                min_confidence = 0.7
            
            if self.confidence_score < min_confidence:
                logger.debug(f"Confidence too low: {self.confidence_score:.2f} < {min_confidence}")
                return
            
            # Balance checks with detailed logging
            usdt_free = self.get_asset_free(self.quote_asset)
            desired_qty = self.calculate_position_size(self.confidence_score)
            required_quote = best_ask * desired_qty * 1.02  # 2% buffer
            
            logger.info(f"Entry conditions met!")
            logger.info(f"  Spread: {spread*100:.3f}% (required: {MIN_SPREAD*100:.3f}%)")
            logger.info(f"  Confidence: {self.confidence_score:.2f}")
            logger.info(f"  Depth ratio: {depth_ratio:.2f}")
            logger.info(f"  Market regime: {self.market_regime}")
            logger.info(f"  Balance check: {usdt_free:.2f} USDT available, {required_quote:.2f} required")
            
            if usdt_free < required_quote:
                logger.warning(f"Insufficient {self.quote_asset} balance: {usdt_free:.2f} < {required_quote:.2f}")
                return

            if not self.notional_ok(best_ask, desired_qty):
                logger.warning(f"NOTIONAL check failed: {best_ask} * {desired_qty} < {self.min_notional}")
                return

            # Store spread for logging
            self.spread_at_entry = spread

            # Place BUY order
            logger.info(f"🚀 PLACING BUY ORDER: {desired_qty} BNB @ ~{best_ask:.4f}")
            entry_price, filled_qty = self.place_order(Client.SIDE_BUY, desired_qty)
            
            if entry_price and filled_qty > 0:
                self.trade_active = True
                self.entry_price = entry_price
                self.position_qty = self.format_quantity(filled_qty)
                self.highest_price_since_entry = entry_price

                self.target_price = self.format_price(entry_price * (1 + PROFIT_TARGET))
                self.stop_price = self.format_price(entry_price * (1 - STOP_LOSS))
                self.original_stop_price = self.stop_price

                msg = (
                    f"📈 BNB ENTRY EXECUTED\n"
                    f"💰 Price: {self.entry_price:.4f}\n"
                    f"🔢 Qty: {self.position_qty}\n"
                    f"🎯 Target: {self.target_price:.4f} (+{PROFIT_TARGET*100:.2f}%)\n"
                    f"🛑 Stop: {self.stop_price:.4f} (-{STOP_LOSS*100:.2f}%)\n"
                    f"📊 Confidence: {self.confidence_score:.1%}\n"
                    f"🌡️ Regime: {self.market_regime}\n"
                    f"📈 Spread: {spread:.4%}\n"
                    f"⭐ Step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}"
                )
                logger.info(msg.replace('\n', ' | '))
                self.send_telegram(msg)
            else:
                logger.error("❌ Order placement failed!")

        except Exception as e:
            logger.error(f"Error in enhanced depth handler: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        def handle_trade_data(self, data):
            """Enhanced trade stream handler with trailing stops"""
            try:
                price = float(data["p"])
                self.current_price = price
                self.price_history.append(price)

                if not self.trade_active:
                    return

                # Update trailing stop
                self.update_trailing_stop()

                take_profit = price >= self.target_price
                stop_loss = price <= self.stop_price

                if not (take_profit or stop_loss):
                    return

                # Enhanced exit logic
                base_free = self.get_asset_free(self.base_asset)
                sellable = min(self.position_qty, base_free) * 0.995
                sell_qty = self.format_quantity(sellable)

                min_check = self.market_min_qty if self.market_min_qty else self.min_qty
                if min_check is not None and sell_qty < min_check:
                    logger.warning(f"Sell qty {sell_qty} below market minQty {min_check}")
                    sell_qty = self.format_quantity(base_free * 0.99)
                    if sell_qty < min_check:
                        logger.error(f"Cannot sell: available {base_free} results in qty {sell_qty} < minQty {min_check}")
                        return

                if self.min_notional and (price * sell_qty) < self.min_notional:
                    logger.error(f"Sell would violate MIN_NOTIONAL: {price * sell_qty:.8f} < {self.min_notional}")
                    return

                exit_price, exec_qty = self.place_order(Client.SIDE_SELL, sell_qty)
                if exit_price and exec_qty > 0:
                    # ORIGINAL calculation (before fees)
                    gross_pct = ((exit_price - self.entry_price) / self.entry_price) * 100.0
                    
                    # NEW: Calculate fees (estimate 0.2% round trip)
                    fee_rate = 0.002  # 0.2% for round trip
                    
                    # Check if using BNB discount
                    try:
                        bnb_balance = self.get_asset_free('BNB')
                        if bnb_balance >= 0.1:
                            fee_rate = 0.0015  # 25% discount = 0.15% round trip
                            logger.info("Using BNB fee discount")
                    except:
                        pass
                    
                    # Calculate net P&L after fees
                    net_pct = gross_pct - (fee_rate * 100)
                    
                    # Use net P&L for logging and display
                    pct = net_pct
                    
                    # Determine exit reason
                    if take_profit:
                        reason = 'profit_target'
                    elif price <= self.original_stop_price:
                        reason = 'stop_loss'
                    else:
                        reason = 'trailing_stop'
                    
                    # Log the trade
                    try:
                        self.trade_logger.log_trade(
                            symbol=SYMBOL,
                            side='BUY',
                            entry_price=self.entry_price,
                            exit_price=exit_price,
                            quantity=self.position_qty,
                            pnl_pct=pct / 100.0,
                            martingale_step=self.martingale_step,
                            reason=reason,
                            spread_at_entry=self.spread_at_entry
                        )
                    except Exception as log_error:
                        logger.error(f"Failed to log trade: {log_error}")
                    
                    # Enhanced exit message
                    if take_profit:
                        msg = f"🎯 PROFIT TARGET HIT!"
                    elif reason == 'trailing_stop':
                        msg = f"📈 TRAILING STOP HIT!"
                    else:
                        msg = f"🛑 STOP LOSS HIT!"
                    
                    msg += (f"\n💰 Sold @ {exit_price:.6f}"
                        f"\n📊 P/L: {pct:.2f}%"
                        f"\n🎯 Confidence was: {self.confidence_score:.1%}"
                        f"\n🌡️ Market: {self.market_regime}")
                    
                    try:
                        if take_profit:
                            self.apply_martingale('profit')
                        else:
                            self.apply_martingale('loss')
                    except Exception as e:
                        logger.error(f"Post-sell handling error: {e}")
                    finally:
                        logger.info(msg.replace('\n', ' | '))
                        self.send_telegram(msg)
                        self.reset_trade_state()
                        
                        # Dynamic cooldown based on outcome
                        if take_profit:
                            self.set_cooldown(COOLDOWN_SECONDS // 2)  # Shorter cooldown after profit
                        else:
                            self.set_cooldown(COOLDOWN_SECONDS)  # Normal cooldown after loss

            except Exception as e:
                logger.error(f"Error in enhanced trade handler: {e}")

        # ---------- All other methods remain the same ----------
        
        def now(self):
            return time.time()

        def in_cooldown(self):
            return self.now() < self.cooldown_until

        def set_cooldown(self, seconds=None):
            if seconds is None:
                seconds = COOLDOWN_SECONDS
            if seconds > 0:
                self.cooldown_until = self.now() + seconds

        def notional_ok(self, price, qty):
            if self.min_notional is None:
                return True
            return (price * qty) >= self.min_notional

        def get_asset_free(self, asset):
            try:
                bal = self.client.get_asset_balance(asset=asset)
                if bal and 'free' in bal:
                    return float(bal['free'])
            except Exception:
                pass
            return self.get_account_balance(asset)

        def format_quantity(self, quantity):
            q = Decimal(str(quantity))
            fmt = Decimal('0.' + '0' * self.quantity_precision) if self.quantity_precision > 0 else Decimal('0')
            return float(q.quantize(fmt, rounding=ROUND_DOWN))

        def format_price(self, price):
            p = Decimal(str(price))
            fmt = Decimal('0.' + '0' * self.price_precision) if self.price_precision > 0 else Decimal('0')
            return float(p.quantize(fmt, rounding=ROUND_DOWN))

        def send_telegram(self, msg):
            if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
                logger.info(f"[Telegram Disabled] {msg}")
                return

            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
            try:
                response = requests.post(url, data=payload, timeout=5)
                if response.status_code == 200:
                    logger.info(f"[Telegram Sent] {msg}")
                else:
                    logger.warning(f"[Telegram Failed] Status: {response.status_code}")
            except Exception as e:
                logger.error(f"[Telegram Error] {e}")

        def get_account_balance(self, asset="USDT"):
            try:
                account = self.client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        return float(balance['free'])
                return 0.0
            except Exception as e:
                logger.error(f"Failed to get balance: {e}")
                return 0.0

        def place_order(self, side, quantity):
            try:
                formatted_qty = self.format_quantity(quantity)

                min_check = self.market_min_qty if self.market_min_qty else self.min_qty
                max_check = self.market_max_qty if self.market_max_qty else self.max_qty

                if min_check is not None and formatted_qty < min_check:
                    logger.error(f"Quantity {formatted_qty} below minimum {min_check}")
                    return (None, 0.0)
                
                if max_check is not None and formatted_qty > max_check:
                    logger.error(f"Quantity {formatted_qty} above maximum {max_check}")
                    return (None, 0.0)

                logger.info(f"Placing {side} order: {formatted_qty} {SYMBOL}")

                order = self.client.order_market(
                    symbol=SYMBOL,
                    side=side,
                    quantity=formatted_qty
                )

                total_qty = 0.0
                total_cost = 0.0
                for fill in order.get("fills", []):
                    fq = float(fill["qty"])
                    fp = float(fill["price"])
                    total_qty += fq
                    total_cost += fq * fp

                executed_qty = float(order.get("executedQty", total_qty) or 0.0)
                avg_price = (total_cost / total_qty) if total_qty > 0 else float(order.get("price", 0.0))

                logger.info(f"Order filled: {side} {executed_qty} {SYMBOL} @ {avg_price:.6f}")
                self.send_telegram(f"✅ {side} {executed_qty} {SYMBOL} @ {avg_price:.6f}")

                return (avg_price, executed_qty)

            except Exception as e:
                error_msg = f"Order error ({side}): {str(e)}"
                logger.error(error_msg)
                self.send_telegram(f"❌ {error_msg}")
                return (None, 0.0)

        def reset_trade_state(self):
            self.trade_active = False
            self.entry_price = 0.0
            self.target_price = 0.0
            self.stop_price = 0.0
            self.position_qty = 0.0
            self.original_stop_price = 0.0
            self.highest_price_since_entry = 0.0
            self.confidence_score = 0.0

        def apply_martingale(self, outcome: str):
            if not MARTINGALE_ENABLED:
                self.current_qty = self.base_qty
                self.martingale_step = 0
                return

            if outcome == 'profit':
                self.current_qty = self.base_qty
                self.martingale_step = 0
            else:
                if self.martingale_step < MARTINGALE_MAX_STEPS:
                    self.martingale_step += 1
                    self.current_qty = self.base_qty * (MARTINGALE_MULTIPLIER ** self.martingale_step)
                else:
                    cap_msg = (f"🛑 Martingale cap reached (step={self.martingale_step}). "
                            f"{'Trading paused.' if HALT_ON_MAX_STEPS else f'Extended cooldown {EXTENDED_COOLDOWN_SECONDS}s.'}")
                    logger.warning(cap_msg)
                    self.send_telegram(cap_msg)
                    if HALT_ON_MAX_STEPS:
                        self.trading_paused = True
                    else:
                        self.set_cooldown(EXTENDED_COOLDOWN_SECONDS)

            logger.info(f"[Martingale] outcome={outcome} step={self.martingale_step} next_qty={self.current_qty}")

        def _get_symbol_info(self):
                """Get symbol information from exchange"""
                try:
                    exchange_info = self.client.get_exchange_info()
                    for symbol in exchange_info['symbols']:
                        if symbol['symbol'] == SYMBOL:
                            self.symbol_info = symbol
                            self.base_asset = symbol['baseAsset']
                            self.quote_asset = symbol['quoteAsset']
                            self.min_notional = None
                            self.max_qty = None
                            self.market_min_qty = None
                            self.market_max_qty = None
                            self.max_num_orders = None

                            for f in symbol['filters']:
                                ftype = f['filterType']
                                if ftype == 'PRICE_FILTER':
                                    tick_size = f['tickSize']
                                    self.price_precision = len(tick_size.rstrip('0').split('.')[1]) if '.' in tick_size else 0
                                elif ftype == 'LOT_SIZE':
                                    step_size = f['stepSize']
                                    self.quantity_precision = len(step_size.rstrip('0').split('.')[1]) if '.' in step_size else 0
                                    self.min_qty = float(f['minQty'])
                                    self.max_qty = float(f.get('maxQty', 0)) if f.get('maxQty') != '9000000000' else None
                                elif ftype == 'MARKET_LOT_SIZE':
                                    self.market_min_qty = float(f['minQty'])
                                    self.market_max_qty = float(f.get('maxQty', 0)) if f.get('maxQty') != '9000000000' else None
                                elif ftype in ('MIN_NOTIONAL', 'NOTIONAL'):
                                    self.min_notional = float(f.get('minNotional') or f.get('notional') or 0.0)
                                elif ftype == 'MAX_NUM_ORDERS':
                                    self.max_num_orders = int(f.get('maxNumOrders', 200))

                            logger.info(
                                f"Enhanced Symbol info: {SYMBOL} | "
                                f"price_precision={self.price_precision}, qty_precision={self.quantity_precision}, "
                                f"min_qty={self.min_qty}, min_notional={self.min_notional}"
                            )
                            return  # Successfully found and configured symbol
                    
                    # If we get here, symbol wasn't found
                    logger.error(f"Symbol {SYMBOL} not found in exchange info")
                    raise Exception(f"Symbol {SYMBOL} not found")
                    
                except Exception as e:
                    logger.error(f"Failed to get symbol info: {e}")
                    # Set default values to prevent crashes
                    self.symbol_info = None
                    self.base_asset = SYMBOL.replace('USDT', '')
                    self.quote_asset = 'USDT'
                    self.price_precision = 4
                    self.quantity_precision = 5
                    self.min_qty = 0.001
                    self.min_notional = 5.0
                    raise e
                    
            # WebSocket methods remain the same as original...
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        self.send_telegram(f"⚠️ WebSocket error: {error}")

    def on_message(self, ws, message):
        """Handle WebSocket messages using the combined stream format"""
        try:
            data = json.loads(message)

            # Combined stream format
            stream = data.get("stream")
            payload = data.get("data")

            if stream and payload:
                if "depth" in stream:
                    self.handle_depth_data(payload)
                elif "trade" in stream:
                    self.handle_trade_data(payload)

            # Fallback: direct format
            elif 'b' in data and 'a' in data:
                self.handle_depth_data(data)
            elif 'p' in data and 's' in data:
                self.handle_trade_data(data)

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def on_depth_message_direct(self, ws, message):
        """Handle direct depth WebSocket messages (for testnet)"""
        try:
            data = json.loads(message)
            self.handle_depth_data(data)
        except Exception as e:
            logger.error(f"Error processing direct depth message: {e}")

    def on_trade_message_direct(self, ws, message):
        """Handle direct trade WebSocket messages (for testnet)"""
        try:
            data = json.loads(message)
            self.handle_trade_data(data)
        except Exception as e:
            logger.error(f"Error processing direct trade message: {e}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        logger.info("WebSocket connection opened successfully")

    def start_websocket_streams(self):
        """Start WebSocket streams with correct URLs for testnet and mainnet"""
        try:
            symbol_lower = SYMBOL.lower()

            if TESTNET:
                # Testnet uses individual stream endpoints, not combined streams
                depth_url = f"wss://stream.testnet.binance.vision/ws/{symbol_lower}@depth@100ms"
                trade_url = f"wss://stream.testnet.binance.vision/ws/{symbol_lower}@trade"

                logger.info("Starting testnet WebSocket streams:")
                logger.info(f"Depth: {depth_url}")
                logger.info(f"Trade: {trade_url}")

                # Create separate WebSocket connections for testnet
                self.depth_ws = websocket.WebSocketApp(
                    depth_url,
                    on_message=self.on_depth_message_direct,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )

                self.trade_ws = websocket.WebSocketApp(
                    trade_url,
                    on_message=self.on_trade_message_direct,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )

                # Start both connections
                depth_thread = threading.Thread(target=self.depth_ws.run_forever, daemon=True)
                depth_thread.start()

                trade_thread = threading.Thread(target=self.trade_ws.run_forever, daemon=True)
                trade_thread.start()

            else:
                # Mainnet uses combined stream format
                streams = [
                    f"{symbol_lower}@depth@100ms",
                    f"{symbol_lower}@trade"
                ]
                stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

                logger.info(f"Starting mainnet WebSocket stream: {stream_url}")

                self.ws = websocket.WebSocketApp(
                    stream_url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )

                ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
                ws_thread.start()

            logger.info("Enhanced WebSocket streams started successfully")
            time.sleep(2)

        except Exception as e:
            logger.error(f"Failed to start WebSocket streams: {e}")
            raise

    def stop_websocket_streams(self):
        """Stop WebSocket streams"""
        try:
            if self.ws:
                self.ws.close()
            if self.depth_ws:
                self.depth_ws.close()
            if self.trade_ws:
                self.trade_ws.close()
            logger.info("WebSocket streams stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket streams: {e}")

    def run(self):
        logger.info("=" * 60)
        logger.info("ENHANCED BINANCE SCALPER WITH WIN RATE OPTIMIZATION")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Base Qty: {QTY}")
        logger.info(f"Min Spread: {MIN_SPREAD}")
        logger.info(f"Profit Target: {PROFIT_TARGET}")
        logger.info(f"Stop Loss: {STOP_LOSS}")
        logger.info(f"Testnet: {TESTNET}")
        logger.info("=" * 30)
        logger.info("WIN RATE ENHANCEMENTS:")
        logger.info(f"✅ Order Book Depth Analysis: {MIN_BOOK_DEPTH}")
        logger.info(f"✅ Spread Stability Check: {MAX_SPREAD_VOLATILITY}")
        logger.info(f"✅ Market Regime Detection: {ENABLE_MARKET_REGIME}")
        logger.info(f"✅ Time-of-Day Filter: {ENABLE_TIME_FILTER}")
        logger.info(f"✅ Trailing Stop Loss: {ENABLE_TRAILING_STOP}")
        logger.info(f"✅ Confidence-Based Sizing: Enabled")
        logger.info(f"✅ Enhanced Cooldowns: {COOLDOWN_SECONDS}s")
        logger.info(f"✅ Reduced Martingale: {MARTINGALE_ENABLED} (max {MARTINGALE_MAX_STEPS})")
        logger.info("=" * 60)

        startup_msg = (
            f"🚀 Enhanced Scalper Started!\n"
            f"📊 {SYMBOL} | Qty: {QTY}\n"
            f"🎯 TP: {PROFIT_TARGET*100:.1f}% | SL: {STOP_LOSS*100:.1f}%\n"
            f"📈 Spread: {MIN_SPREAD*100:.2f}% | Depth: {MIN_BOOK_DEPTH}\n"
            f"🧠 AI Features: Market Regime + Confidence Scoring\n"
            f"⏰ Time Filter + Trailing Stops Enabled\n"
            f"🎲 Martingale: {'ON' if MARTINGALE_ENABLED else 'OFF'}"
        )
        self.send_telegram(startup_msg)

        try:
            self.start_websocket_streams()

            last_status_time = time.time()
            last_regime_log = ""
            
            while True:
                time.sleep(1)

                # Enhanced heartbeat every 5 minutes
                current_time = time.time()
                if current_time - last_status_time > 300:
                    status = 'Trading' if self.trade_active else 'Monitoring'
                    if self.trading_paused:
                        status += " (PAUSED)"
                    elif not self.trade_active and self.in_cooldown():
                        status += f" (cooldown {int(self.cooldown_until - self.now())}s)"

                    # Enhanced status with market analysis
                    regime_info = f"Regime:{self.market_regime}"
                    if self.market_regime != last_regime_log:
                        logger.info(f"🌡️ Market regime changed: {last_regime_log} -> {self.market_regime}")
                        last_regime_log = self.market_regime

                    volatility = self.calculate_recent_volatility()
                    spread_avg = np.mean(list(self.spread_history)[-20:]) if len(self.spread_history) >= 20 else 0
                    
                    status_msg = (
                        f"🔄 Status: {status} | {regime_info}\n"
                        f"📊 Vol: {volatility*100:.2f}% | Spread: {spread_avg*100:.3f}%\n"
                        f"🎯 Confidence: {self.confidence_score:.1%} | Qty: {self.current_qty}\n"
                        f"📈 Price History: {len(self.price_history)} samples"
                    )
                    
                    if self.trade_active:
                        profit_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100
                        status_msg += (
                            f"\n💰 Active Trade: Entry {self.entry_price:.6f} | "
                            f"Current {self.current_price:.6f} | P/L: {profit_pct:.2f}%"
                        )
                    
                    logger.info(status_msg.replace('\n', ' | '))
                    
                    # Send extended status to Telegram every hour
                    if current_time - last_status_time > 3600:  # 1 hour
                        self.send_telegram(status_msg)
                    
                    last_status_time = current_time

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.send_telegram(f"❌ Enhanced Scalper error: {e}")

        finally:
            self.stop_websocket_streams()

            if self.trade_active:
                warning_msg = (f"⚠️ Enhanced Scalper stopped with active trade!\n"
                            f"Entry: {self.entry_price:.6f} | Current: {self.current_price:.6f}")
                logger.warning(warning_msg.replace('\n', ' | '))
                self.send_telegram(warning_msg)

            logger.info("Enhanced Scalper stopped")
            self.send_telegram("🛑 Enhanced Scalper Stopped")

    def calculate_net_profit_after_fees(self, entry_price, exit_price, quantity, side='BUY'):
        """Calculate actual profit after transaction fees"""
        
        # Determine effective fee rates
        effective_maker_fee = MAKER_FEE * (1 - BNB_FEE_DISCOUNT) if self.has_bnb_for_fees() else MAKER_FEE
        effective_taker_fee = TAKER_FEE * (1 - BNB_FEE_DISCOUNT) if self.has_bnb_for_fees() else TAKER_FEE
        
        if side == 'BUY':
            # Buy at entry_price (market order = taker fee)
            # Sell at exit_price (market order = taker fee)
            buy_fee = entry_price * quantity * effective_taker_fee
            sell_fee = exit_price * quantity * effective_taker_fee
            
            gross_pnl = (exit_price - entry_price) * quantity
            net_pnl = gross_pnl - buy_fee - sell_fee
            
            # Calculate percentage return on investment
            total_investment = entry_price * quantity + buy_fee
            net_pnl_pct = net_pnl / total_investment
            
        else:  # SHORT position
            buy_fee = exit_price * quantity * effective_taker_fee
            sell_fee = entry_price * quantity * effective_taker_fee
            
            gross_pnl = (entry_price - exit_price) * quantity
            net_pnl = gross_pnl - buy_fee - sell_fee
            
            total_investment = entry_price * quantity + sell_fee
            net_pnl_pct = net_pnl / total_investment
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'net_pnl_pct': net_pnl_pct,
            'total_fees': buy_fee + sell_fee,
            'buy_fee': buy_fee,
            'sell_fee': sell_fee,
            'effective_fee_rate': (effective_maker_fee + effective_taker_fee) / 2
        }

    def has_bnb_for_fees(self):
        """Check if account has sufficient BNB for fee discounts"""
        try:
            bnb_balance = self.get_asset_free('BNB')
            # Need at least 0.1 BNB for fee payments
            return bnb_balance >= 0.1
        except:
            return False

    def calculate_minimum_profitable_spread(self):
        """Calculate minimum spread needed to be profitable after fees"""
        # Round-trip fees
        total_fee_rate = (MAKER_FEE + TAKER_FEE) * (1 - BNB_FEE_DISCOUNT if self.has_bnb_for_fees() else 1)
        
        # Add profit buffer (minimum 0.1% profit after fees)
        min_profit_buffer = 0.001  # 0.1%
        
        # Minimum spread = fees + profit buffer + slippage buffer
        min_spread = total_fee_rate + min_profit_buffer + 0.0005  # 0.05% slippage
        
        return min_spread

    def adjust_targets_for_fees(self, base_profit_target, base_stop_loss):
        """Adjust profit targets and stop losses to account for fees"""
        
        total_fee_rate = (MAKER_FEE + TAKER_FEE) * (1 - BNB_FEE_DISCOUNT if self.has_bnb_for_fees() else 1)
        
        # Adjust profit target to achieve desired net profit
        adjusted_profit_target = base_profit_target + total_fee_rate + 0.0002  # Small buffer
        
        # Adjust stop loss to limit total loss including fees  
        adjusted_stop_loss = base_stop_loss + total_fee_rate * 0.5  # Partial fee adjustment
        
        return adjusted_profit_target, adjusted_stop_loss

    # Enhanced order placement with fee awareness
def place_order(self, side, quantity):
    """Place market order with fee-aware error handling"""
    try:
        formatted_qty = self.format_quantity(quantity)
        
        # EXISTING VALIDATION CODE (keep as is)
        min_check = self.market_min_qty if self.market_min_qty else self.min_qty
        max_check = self.market_max_qty if self.market_max_qty else self.max_qty

        if min_check is not None and formatted_qty < min_check:
            logger.error(f"Quantity {formatted_qty} below minimum {min_check}")
            return (None, 0.0)
        
        if max_check is not None and formatted_qty > max_check:
            logger.error(f"Quantity {formatted_qty} above maximum {max_check}")
            return (None, 0.0)

        logger.info(f"Placing {side} order: {formatted_qty} {SYMBOL}")

        # EXISTING ORDER PLACEMENT (keep as is)
        order = self.client.order_market(
            symbol=SYMBOL,
            side=side,
            quantity=formatted_qty
        )

        # EXISTING FILL PROCESSING (keep as is)
        total_qty = 0.0
        total_cost = 0.0
        for fill in order.get("fills", []):
            fq = float(fill["qty"])
            fp = float(fill["price"])
            total_qty += fq
            total_cost += fq * fp

        executed_qty = float(order.get("executedQty", total_qty) or 0.0)
        avg_price = (total_cost / total_qty) if total_qty > 0 else float(order.get("price", 0.0))

        # ADD THIS: Calculate fees from the order
        total_fees = 0.0
        for fill in order.get("fills", []):
            commission = float(fill.get("commission", 0))
            total_fees += commission

        logger.info(f"Order filled: {side} {executed_qty} {SYMBOL} @ {avg_price:.6f} (fees: ${total_fees:.4f})")
        self.send_telegram(f"✅ {side} {executed_qty} {SYMBOL} @ {avg_price:.6f}")

        return (avg_price, executed_qty)

    except Exception as e:
        error_msg = f"Order error ({side}): {str(e)}"
        logger.error(error_msg)
        self.send_telegram(f"❌ {error_msg}")
        return (None, 0.0)

    # Enhanced trade logging with fee calculation
    def log_trade_with_fees(self, entry_price, exit_price, quantity, reason):
        """Log trade with accurate fee-adjusted P&L"""
        
        fee_analysis = self.calculate_net_profit_after_fees(entry_price, exit_price, quantity)
        
        try:
            self.trade_logger.log_trade(
                symbol=SYMBOL,
                side='BUY',
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl_pct=fee_analysis['net_pnl_pct'],  # Use NET P&L after fees
                martingale_step=self.martingale_step,
                reason=reason,
                spread_at_entry=self.spread_at_entry,
                # Add fee information to market_conditions
                market_conditions=json.dumps({
                    'gross_pnl_pct': fee_analysis['gross_pnl'] / (entry_price * quantity),
                    'net_pnl_pct': fee_analysis['net_pnl_pct'],
                    'total_fees': fee_analysis['total_fees'],
                    'fee_rate_used': fee_analysis['effective_fee_rate'],
                    'bnb_discount_applied': self.has_bnb_for_fees()
                })
            )
            
            # Enhanced telegram message with fee info
            net_pnl_pct = fee_analysis['net_pnl_pct'] * 100
            gross_pnl_pct = fee_analysis['gross_pnl'] / (entry_price * quantity) * 100
            total_fees_usd = fee_analysis['total_fees']
            
            fee_msg = (
                f"💰 Gross P&L: {gross_pnl_pct:.2f}%\n"
                f"💸 Fees Paid: ${total_fees_usd:.4f}\n"
                f"📊 Net P&L: {net_pnl_pct:.2f}%\n"
                f"🎫 BNB Discount: {'✅' if self.has_bnb_for_fees() else '❌'}"
            )
            
            self.send_telegram(fee_msg)
            
        except Exception as e:
            logger.error(f"Failed to log trade with fees: {e}")


    def test_fee_calculation(self):
        """Test fee calculation with current settings"""
        entry_price = 100.0
        exit_price = 100.3  # 0.3% profit
        quantity = 1.0
        
        gross_profit = (exit_price - entry_price) / entry_price
        fees = 0.002  # 0.2% round trip
        net_profit = gross_profit - fees
        
        logger.info(f"Fee Test: Gross {gross_profit:.4f} - Fees {fees:.4f} = Net {net_profit:.4f}")
        return net_profit

    def main():
        """Main function with enhanced validation"""
        if not API_KEY or not API_SECRET:
            logger.error("❌ Missing required API credentials")
            return

        # Validate enhanced parameters
        logger.info("🔍 Validating enhanced parameters...")
        
        if MIN_SPREAD < 0.001:
            logger.warning(f"⚠️ MIN_SPREAD very low: {MIN_SPREAD}")
        if PROFIT_TARGET <= STOP_LOSS:
            logger.warning(f"⚠️ Risk/Reward ratio concerning: TP={PROFIT_TARGET} SL={STOP_LOSS}")
        if MIN_BOOK_DEPTH < 100:
            logger.warning(f"⚠️ MIN_BOOK_DEPTH very low: {MIN_BOOK_DEPTH}")

        try:
            scalper = EnhancedBinanceScalper()
            scalper.run()
        except Exception as e:
            logger.error(f"Failed to start enhanced scalper: {e}")


if __name__ == "__main__":
    main()
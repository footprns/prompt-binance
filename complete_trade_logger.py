#!/usr/bin/env python3
"""
Trade Logger Integration for Binance Scalper
This file contains the TradeLogger class and integration instructions
"""

import sqlite3
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradeLogger:
    """Logs trades to database for AI analysis"""
    
    def __init__(self, db_path="/home/ubuntu/binance_scalper/trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl_pct REAL,
                    pnl_usd REAL,
                    martingale_step INTEGER,
                    reason TEXT,
                    spread_at_entry REAL,
                    market_conditions TEXT
                )
            ''')
            
            # Create index for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Trade logging database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trade database: {e}")
    
    def log_trade(self, symbol, side, entry_price, exit_price, quantity, 
                  pnl_pct, martingale_step, reason, spread_at_entry=0.0, 
                  additional_data=None):
        """Log completed trade"""
        
        try:
            pnl_usd = (exit_price - entry_price) * quantity if side == 'BUY' else (entry_price - exit_price) * quantity
            
            # Get market conditions (you can expand this)
            market_conditions = {
                'timestamp': datetime.now().isoformat(),
                'volatility': 'normal',  # You can calculate this
                'volume': 'normal',      # You can get this from API
                'trend': 'unknown',      # You can implement trend detection
                'session': self._get_trading_session(),
                'additional': additional_data or {}
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (timestamp, symbol, side, entry_price, exit_price, quantity, 
                 pnl_pct, pnl_usd, martingale_step, reason, spread_at_entry, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                side,
                entry_price,
                exit_price,
                quantity,
                pnl_pct,
                pnl_usd,
                martingale_step,
                reason,
                spread_at_entry,
                json.dumps(market_conditions)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Trade logged: {symbol} {side} {pnl_pct:.3%} P&L")
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def _get_trading_session(self):
        """Determine trading session (Asian, European, US)"""
        hour = datetime.now().hour
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'european'
        else:
            return 'us'
    
    def get_trade_count(self, symbol, hours=24):
        """Get trade count for last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
            '''.format(hours), (symbol,))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get trade count: {e}")
            return 0

"""
INTEGRATION INSTRUCTIONS:

1. Add the following imports to the top of your scalper_ws_full.py:
   from trade_logger_integration import TradeLogger

2. Add to your BinanceScalper.__init__() method:
   # Add trade logger
   self.trade_logger = TradeLogger()
   self.spread_at_entry = 0.0  # Track spread when entering

3. Modify your handle_depth_data method to track spread:
   In the section where you calculate spread, add:
   # Store spread for logging
   self.spread_at_entry = spread

4. Modify your handle_trade_data method to log trades:
   After the successful sell order, add the logging call.

Here are the exact code snippets to add:
"""

# ========================================
# CODE TO ADD TO YOUR EXISTING scalper_ws_full.py
# ========================================

"""
1. AT THE TOP OF scalper_ws_full.py, add this import:
"""
# from trade_logger_integration import TradeLogger

"""
2. IN THE BinanceScalper.__init__() method, add these lines after symbol info initialization:
"""
# # Add trade logger
# self.trade_logger = TradeLogger()
# self.spread_at_entry = 0.0  # Track spread when entering

"""
3. IN THE handle_depth_data method, after calculating spread, add:
"""
# # Store spread for logging
# self.spread_at_entry = spread

"""
4. IN THE handle_trade_data method, after the successful sell order and before reset_trade_state(), add:
"""
# # Log the trade to database
# reason = 'profit_target' if take_profit else 'stop_loss'
# self.trade_logger.log_trade(
#     symbol=SYMBOL,
#     side='BUY',  # We always buy first in scalping
#     entry_price=self.entry_price,
#     exit_price=exit_price,
#     quantity=self.position_qty,
#     pnl_pct=pct / 100.0,  # Convert to decimal
#     martingale_step=self.martingale_step,
#     reason=reason,
#     spread_at_entry=self.spread_at_entry
# )

"""
COMPLETE MODIFIED METHODS:
Here are the complete modified methods for your reference:
"""

def modified_handle_depth_data(self, data):
    """Handle depth stream data for entry signals - MODIFIED VERSION"""
    # Don't try to re-enter while in a position, during cooldown, or when paused
    if self.trade_active or self.in_cooldown() or self.trading_paused:
        return

    try:
        bids = data.get("b", [])
        asks = data.get("a", [])
        if not bids or not asks:
            return

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = (best_ask - best_bid) / best_bid

        # Update current price mid
        self.current_price = (best_bid + best_ask) / 2.0
        
        # Store spread for logging (NEW LINE)
        self.spread_at_entry = spread

        if spread >= MIN_SPREAD:
            logger.info(f"Spread trigger: {spread:.6f} >= {MIN_SPREAD}")

            # Use martingale-adjusted size
            desired_qty = self.current_qty

            # Check balances & notional
            usdt_free = self.get_asset_free(self.quote_asset)
            required_quote = best_ask * desired_qty * 1.02  # 2% buffer for fees/slippage
            if usdt_free < required_quote:
                logger.warning(f"Insufficient {self.quote_asset} balance: {usdt_free} < {required_quote}")
                return

            if not self.notional_ok(best_ask, desired_qty):
                logger.warning(f"NOTIONAL check failed: {best_ask} * {desired_qty} < {self.min_notional}")
                return

            # Place BUY
            entry_price, filled_qty = self.place_order(Client.SIDE_BUY, desired_qty)
            if entry_price and filled_qty > 0:
                self.trade_active = True
                self.entry_price = entry_price
                self.position_qty = self.format_quantity(filled_qty)

                self.target_price = self.format_price(entry_price * (1 + PROFIT_TARGET))
                self.stop_price = self.format_price(entry_price * (1 - STOP_LOSS))

                msg = (
                    f"📈 ENTRY: {self.entry_price:.6f}\n"
                    f"🔢 Qty: {self.position_qty}\n"
                    f"🎯 TP: {self.target_price:.6f}\n"
                    f"🛑 SL: {self.stop_price:.6f}\n"
                    f"🎛️ Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}"
                )
                logger.info(msg.replace('\n', ' | '))
                self.send_telegram(msg)

    except Exception as e:
        logger.error(f"Error in depth handler: {e}")

def modified_handle_trade_data(self, data):
    """Handle trade stream data for exit signals - MODIFIED VERSION"""
    try:
        price = float(data["p"])
        self.current_price = price

        if not self.trade_active:
            return

        take_profit = price >= self.target_price
        stop_loss = price <= self.stop_price

        if not (take_profit or stop_loss):
            return

        # Sell only what we truly hold, with more conservative buffer
        base_free = self.get_asset_free(self.base_asset)
        sellable = min(self.position_qty, base_free) * 0.995  # 99.5% to account for fees
        sell_qty = self.format_quantity(sellable)

        # Use market-specific minimum quantity if available
        min_check = self.market_min_qty if self.market_min_qty else self.min_qty
        if min_check is not None and sell_qty < min_check:
            logger.warning(f"Sell qty {sell_qty} below market minQty {min_check}; attempting adjusted calculation.")
            # Try using 99% of available balance
            sell_qty = self.format_quantity(base_free * 0.99)
            if sell_qty < min_check:
                logger.error(f"Cannot sell: available {base_free} results in qty {sell_qty} < minQty {min_check}")
                return

        # Ensure sell notional is valid to avoid -1013
        if self.min_notional and (price * sell_qty) < self.min_notional:
            logger.error(f"Sell would violate MIN_NOTIONAL: {price * sell_qty:.8f} < {self.min_notional}")
            return

        exit_price, exec_qty = self.place_order(Client.SIDE_SELL, sell_qty)
        if exit_price and exec_qty > 0:
            pct = ((exit_price - self.entry_price) / self.entry_price) * 100.0
            
            # Log the trade to database (NEW SECTION)
            reason = 'profit_target' if take_profit else 'stop_loss'
            self.trade_logger.log_trade(
                symbol=SYMBOL,
                side='BUY',  # We always buy first in scalping
                entry_price=self.entry_price,
                exit_price=exit_price,
                quantity=self.position_qty,
                pnl_pct=pct / 100.0,  # Convert to decimal
                martingale_step=self.martingale_step,
                reason=reason,
                spread_at_entry=self.spread_at_entry
            )
            
            msg = ""
            try:
                if take_profit:
                    msg = f"🎯 PROFIT TARGET HIT!\nSold @ {exit_price:.6f}\nP/L: {pct:.2f}%"
                    self.apply_martingale('profit')
                else:
                    msg = f"🛑 STOP LOSS HIT!\nSold @ {exit_price:.6f}\nP/L: {pct:.2f}%"
                    self.apply_martingale('loss')
            except Exception as e:
                logger.error(f"Post-sell handling error: {e}")
                if not msg:
                    msg = f"✅ SELL done @ {exit_price:.6f}\nP/L: {pct:.2f}%"
            finally:
                logger.info(msg.replace('\n', ' | '))
                self.send_telegram(msg)
                self.reset_trade_state()
                self.set_cooldown()

    except Exception as e:
        logger.error(f"Error in trade handler: {e}")

"""
INTEGRATION STEPS:

1. Save this file as trade_logger_integration.py in your scalper directory
2. Add the import to your scalper_ws_full.py
3. Add the initialization lines to __init__
4. Add the spread tracking line to handle_depth_data
5. Add the trade logging section to handle_trade_data
6. Test with your existing bot to ensure logging works

The TradeLogger will create a SQLite database and start logging all your trades
for the AI agent to analyze.
"""
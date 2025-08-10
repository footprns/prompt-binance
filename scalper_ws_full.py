import os
import time
import requests
import logging
import json
import websocket
import threading
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from binance.client import Client
from trade_logger_integration import TradeLogger

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
MIN_SPREAD = float(os.getenv("MIN_SPREAD", "0.0015"))
PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", "0.002"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.001"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Cooldown & Martingale
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))

MARTINGALE_ENABLED = os.getenv("MARTINGALE_ENABLED", "true").lower() == "true"
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
MARTINGALE_MAX_STEPS = int(os.getenv("MARTINGALE_MAX_STEPS", "4"))  # base + up to 4 increases

HALT_ON_MAX_STEPS = os.getenv("HALT_ON_MAX_STEPS", "true").lower() == "true"
EXTENDED_COOLDOWN_SECONDS = int(os.getenv("EXTENDED_COOLDOWN_SECONDS", "900"))  # 15m


class BinanceScalper:
    def __init__(self):
        # Initialize Binance client
        self.client = Client(API_KEY, API_SECRET, testnet=TESTNET)

        # WebSocket connections
        self.ws = None  # For mainnet combined stream
        self.depth_ws = None  # For testnet individual streams
        self.trade_ws = None  # For testnet individual streams

        # Trade state
        self.trade_active = False
        self.trading_paused = False
        self.entry_price = 0.0
        self.target_price = 0.0
        self.stop_price = 0.0
        self.current_price = 0.0

        # Executed position qty (from fills), not the configured QTY
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

        # Initialize symbol info
        self._get_symbol_info()

        self.trade_logger = TradeLogger()
        self.spread_at_entry = 0.0

    # ---------- Exchange Metadata ----------
    def _get_symbol_info(self):
        """Get symbol information for proper precision handling"""
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
                        f"Symbol info: {SYMBOL} | base={self.base_asset}, quote={self.quote_asset} | "
                        f"price_precision={self.price_precision}, qty_precision={self.quantity_precision}, "
                        f"min_qty={self.min_qty}, market_min_qty={self.market_min_qty}, "
                        f"min_notional={self.min_notional}, max_orders={self.max_num_orders}"
                    )
                    break
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")

    # ---------- Helpers ----------
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
        """Format quantity according to symbol precision"""
        q = Decimal(str(quantity))
        fmt = Decimal('0.' + '0' * self.quantity_precision) if self.quantity_precision > 0 else Decimal('0')
        return float(q.quantize(fmt, rounding=ROUND_DOWN))

    def format_price(self, price):
        """Format price according to symbol precision"""
        p = Decimal(str(price))
        fmt = Decimal('0.' + '0' * self.price_precision) if self.price_precision > 0 else Decimal('0')
        return float(p.quantize(fmt, rounding=ROUND_DOWN))

    # ---------- Notifications ----------
    def send_telegram(self, msg):
        """Send Telegram notification"""
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

    # ---------- Account ----------
    def get_account_balance(self, asset="USDT"):
        """Get account balance for specified asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    # ---------- Orders ----------
    def place_order(self, side, quantity):
        """Place market order with proper error handling. Returns (avg_price, executed_qty) or (None, 0)."""
        try:
            formatted_qty = self.format_quantity(quantity)

            # Market LOT_SIZE check (preferred for market orders)
            min_check = self.market_min_qty if self.market_min_qty else self.min_qty
            max_check = self.market_max_qty if self.market_max_qty else self.max_qty

            if min_check is not None and formatted_qty < min_check:
                logger.error(f"Quantity {formatted_qty} below minimum {min_check} (market order)")
                return (None, 0.0)
            
            if max_check is not None and formatted_qty > max_check:
                logger.error(f"Quantity {formatted_qty} above maximum {max_check} (market order)")
                return (None, 0.0)

            logger.info(f"Placing {side} order: {formatted_qty} {SYMBOL}")

            order = self.client.order_market(
                symbol=SYMBOL,
                side=side,
                quantity=formatted_qty
            )

            # Executed qty & average price
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

    # ---------- WebSocket callbacks ----------
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


    def handle_depth_data(self, data):
        """Handle depth stream data for entry signals"""
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
            
            # Store spread for logging
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


    def handle_trade_data(self, data):
        """Handle trade stream data for exit signals"""
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
                
                # Log the trade to database
                reason = 'profit_target' if take_profit else 'stop_loss'
                try:
                    self.trade_logger.log_trade(
                        symbol=SYMBOL,
                        side='BUY',
                        entry_price=self.entry_price,
                        exit_price=exit_price,
                        quantity=self.position_qty,
                        pnl_pct=pct / 100.0,  # Convert to decimal
                        martingale_step=self.martingale_step,
                        reason=reason,
                        spread_at_entry=self.spread_at_entry
                    )
                except Exception as log_error:
                    logger.error(f"Failed to log trade: {log_error}")
                
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

    # ---------- State resets ----------
    def reset_trade_state(self):
        """Reset only position-related state (not martingale/cooldown)."""
        self.trade_active = False
        self.entry_price = 0.0
        self.target_price = 0.0
        self.stop_price = 0.0
        self.position_qty = 0.0

    def apply_martingale(self, outcome: str):
        """Update current_qty based on outcome, respecting hard cap."""
        if not MARTINGALE_ENABLED:
            self.current_qty = self.base_qty
            self.martingale_step = 0
            return

        if outcome == 'profit':
            # Reset after a win
            self.current_qty = self.base_qty
            self.martingale_step = 0
        else:
            # Loss path
            if self.martingale_step < MARTINGALE_MAX_STEPS:
                self.martingale_step += 1
                self.current_qty = self.base_qty * (MARTINGALE_MULTIPLIER ** self.martingale_step)
            else:
                # Cap reached
                cap_msg = (f"🛑 Martingale cap reached (step={self.martingale_step}). "
                        f"{'Trading paused.' if HALT_ON_MAX_STEPS else f'Extended cooldown {EXTENDED_COOLDOWN_SECONDS}s.'}")
                logger.warning(cap_msg)
                self.send_telegram(cap_msg)
                if HALT_ON_MAX_STEPS:
                    self.trading_paused = True
                else:
                    self.set_cooldown(EXTENDED_COOLDOWN_SECONDS)

        logger.info(f"[Martingale] outcome={outcome} step={self.martingale_step} next_qty={self.current_qty}")

    # ---------- Streams ----------
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

            logger.info("WebSocket streams started successfully")
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

    # ---------- Main loop ----------
    def run(self):
        logger.info("=" * 50)
        logger.info("BINANCE SCALPER STARTING")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Base Qty: {QTY}")
        logger.info(f"Min Spread: {MIN_SPREAD}")
        logger.info(f"Profit Target: {PROFIT_TARGET}")
        logger.info(f"Stop Loss: {STOP_LOSS}")
        logger.info(f"Testnet: {TESTNET}")
        logger.info("=" * 50)

        startup_msg = (
            f"🚀 Scalper Started!\n"
            f"📊 {SYMBOL} | Base Qty: {QTY}\n"
            f"📈 TP: {PROFIT_TARGET*100:.1f}% | SL: {STOP_LOSS*100:.1f}%\n"
            f"🧮 MG: x{MARTINGALE_MULTIPLIER}, max {MARTINGALE_MAX_STEPS}, "
            f"{'HALT' if HALT_ON_MAX_STEPS else f'EXT_COOLDOWN {EXTENDED_COOLDOWN_SECONDS}s'}"
        )
        self.send_telegram(startup_msg)

        try:
            self.start_websocket_streams()

            last_status_time = time.time()
            while True:
                time.sleep(1)

                # heartbeat every 5 minutes
                current_time = time.time()
                if current_time - last_status_time > 300:
                    status = 'Trading' if self.trade_active else 'Monitoring'
                    if self.trading_paused:
                        status += " (PAUSED)"
                    elif not self.trade_active and self.in_cooldown():
                        status += f" (cooldown {int(self.cooldown_until - self.now())}s)"

                    status_msg = (
                        f"🔄 Status: {status} | Qty:{self.current_qty} | "
                        f"MG-step:{self.martingale_step}/{MARTINGALE_MAX_STEPS}"
                    )
                    if self.trade_active:
                        status_msg += (
                            f" | Entry:{self.entry_price:.6f} | Px:{self.current_price:.6f} | "
                            f"TP:{self.target_price:.6f} | SL:{self.stop_price:.6f}"
                        )
                    logger.info(status_msg)
                    last_status_time = current_time

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.send_telegram(f"❌ Scalper error: {e}")

        finally:
            self.stop_websocket_streams()

            if self.trade_active:
                warning_msg = f"⚠️ Scalper stopped with active trade!\nEntry: {self.entry_price:.6f}"
                logger.warning(warning_msg.replace('\n', ' | '))
                self.send_telegram(warning_msg)

            logger.info("Scalper stopped")
            self.send_telegram("🛑 Scalper Stopped")


def main():
    """Main function"""
    if not API_KEY or not API_SECRET:
        logger.error("Missing required API credentials")
        return

    try:
        scalper = BinanceScalper()
        scalper.run()
    except Exception as e:
        logger.error(f"Failed to start scalper: {e}")


if __name__ == "__main__":
    main()
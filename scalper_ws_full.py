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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
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
        self.entry_price = 0
        self.target_price = 0
        self.stop_price = 0
        self.current_price = 0
        
        # Symbol info for precision
        self.symbol_info = None
        self.price_precision = 8
        self.quantity_precision = 8
        
        # Initialize symbol info
        self._get_symbol_info()
    
    def _get_symbol_info(self):
        """Get symbol information for proper precision handling"""
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == SYMBOL:
                    self.symbol_info = symbol
                    # Get price and quantity precision
                    for filter_item in symbol['filters']:
                        if filter_item['filterType'] == 'PRICE_FILTER':
                            # Calculate precision from tickSize
                            tick_size = filter_item['tickSize']
                            self.price_precision = len(tick_size.rstrip('0').split('.')[1]) if '.' in tick_size else 0
                        elif filter_item['filterType'] == 'LOT_SIZE':
                            # Calculate precision from stepSize
                            step_size = filter_item['stepSize']
                            self.quantity_precision = len(step_size.rstrip('0').split('.')[1]) if '.' in step_size else 0
                    break
            logger.info(f"Symbol info loaded - Price precision: {self.price_precision}, Quantity precision: {self.quantity_precision}")
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
    
    def format_quantity(self, quantity):
        """Format quantity according to symbol precision"""
        return float(Decimal(str(quantity)).quantize(Decimal('0.' + '0' * self.quantity_precision), rounding=ROUND_DOWN))
    
    def format_price(self, price):
        """Format price according to symbol precision"""
        return float(Decimal(str(price)).quantize(Decimal('0.' + '0' * self.price_precision), rounding=ROUND_DOWN))
    
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
    
    def place_order(self, side, quantity):
        """Place market order with proper error handling"""
        try:
            # Format quantity
            formatted_qty = self.format_quantity(quantity)
            
            # Check minimum quantity requirements
            if self.symbol_info:
                for filter_item in self.symbol_info['filters']:
                    if filter_item['filterType'] == 'LOT_SIZE':
                        min_qty = float(filter_item['minQty'])
                        if formatted_qty < min_qty:
                            logger.error(f"Quantity {formatted_qty} below minimum {min_qty}")
                            return None
            
            logger.info(f"Placing {side} order: {formatted_qty} {SYMBOL}")
            
            order = self.client.order_market(
                symbol=SYMBOL,
                side=side,
                quantity=formatted_qty
            )
            
            # Calculate average fill price
            total_qty = 0
            total_cost = 0
            for fill in order["fills"]:
                fill_qty = float(fill["qty"])
                fill_price = float(fill["price"])
                total_qty += fill_qty
                total_cost += fill_qty * fill_price
            
            avg_price = total_cost / total_qty if total_qty > 0 else 0
            
            logger.info(f"Order filled: {side} {total_qty} {SYMBOL} @ {avg_price:.6f}")
            self.send_telegram(f"✅ {side} {total_qty} {SYMBOL} @ {avg_price:.6f}")
            
            return avg_price
            
        except Exception as e:
            error_msg = f"Order error ({side}): {str(e)}"
            logger.error(error_msg)
            self.send_telegram(f"❌ {error_msg}")
            return None
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        self.send_telegram(f"⚠️ WebSocket error: {error}")


    def on_message(self, ws, message):
        """Handle WebSocket messages using the correct format from your working example"""
        try:
            data = json.loads(message)
            
            # Handle combined stream format (like your working example)
            stream = data.get("stream")
            payload = data.get("data")
            
            if stream and payload:
                if "depth" in stream:
                    self.handle_depth_data(payload)
                elif "trade" in stream:
                    self.handle_trade_data(payload)
            
            # Handle direct stream format (fallback)
            elif 'b' in data and 'a' in data:
                self.handle_depth_data(data)
            elif 'p' in data and 's' in data:
                self.handle_trade_data(data)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def handle_depth_data(self, data):
        """Handle depth stream data for entry signals"""
        if self.trade_active:
            return
        
        try:
            bids = data.get("b", [])
            asks = data.get("a", [])
            
            if not bids or not asks:
                return
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Update current price for monitoring
            self.current_price = (best_bid + best_ask) / 2
            
            logger.debug(f"Spread: {spread:.6f} (min: {MIN_SPREAD})")
            
            if spread >= MIN_SPREAD:
                logger.info(f"Spread trigger: {spread:.6f} >= {MIN_SPREAD}")
                
                # Check balance before placing order
                balance = self.get_account_balance("USDT")
                required_balance = best_ask * QTY * 1.1  # 10% buffer
                
                if balance < required_balance:
                    logger.warning(f"Insufficient balance: {balance} < {required_balance}")
                    return
                
                entry_price = self.place_order(Client.SIDE_BUY, QTY)
                
                if entry_price:
                    self.trade_active = True
                    self.entry_price = entry_price
                    self.target_price = self.format_price(entry_price * (1 + PROFIT_TARGET))
                    self.stop_price = self.format_price(entry_price * (1 - STOP_LOSS))
                    
                    msg = f"📈 ENTRY: {self.entry_price:.6f}\n🎯 TP: {self.target_price:.6f}\n🛑 SL: {self.stop_price:.6f}"
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
            
            logger.debug(f"Price: {price:.6f} (TP: {self.target_price:.6f}, SL: {self.stop_price:.6f})")
            
            if price >= self.target_price:
                exit_price = self.place_order(Client.SIDE_SELL, QTY)
                if exit_price:
                    profit = ((exit_price - self.entry_price) / self.entry_price) * 100
                    msg = f"🎯 PROFIT TARGET HIT!\nSold @ {exit_price:.6f}\nProfit: {profit:.2f}%"
                    logger.info(msg.replace('\n', ' | '))
                    self.send_telegram(msg)
                    self.reset_trade_state()
            
            elif price <= self.stop_price:
                exit_price = self.place_order(Client.SIDE_SELL, QTY)
                if exit_price:
                    loss = ((exit_price - self.entry_price) / self.entry_price) * 100
                    msg = f"🛑 STOP LOSS HIT!\nSold @ {exit_price:.6f}\nLoss: {loss:.2f}%"
                    logger.info(msg.replace('\n', ' | '))
                    self.send_telegram(msg)
                    self.reset_trade_state()
        
        except Exception as e:
            logger.error(f"Error in trade handler: {e}")
    
    def on_depth_message_direct(self, ws, message):
        """Handle direct depth WebSocket messages (for testnet)"""
        if self.trade_active:
            return
        
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
        """Handle WebSocket close"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connection opened successfully")
    
    def reset_trade_state(self):
        """Reset trade state variables"""
        self.trade_active = False
        self.entry_price = 0
        self.target_price = 0
        self.stop_price = 0
    
    def start_websocket_streams(self):
        """Start WebSocket streams with correct URLs for testnet and mainnet"""
        try:
            symbol_lower = SYMBOL.lower()
            
            if TESTNET:
                # Testnet uses individual stream endpoints, not combined streams
                depth_url = f"wss://stream.testnet.binance.vision/ws/{symbol_lower}@depth@100ms"
                trade_url = f"wss://stream.testnet.binance.vision/ws/{symbol_lower}@trade"

                
                logger.info(f"Starting testnet WebSocket streams:")
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
                depth_thread = threading.Thread(target=self.depth_ws.run_forever)
                depth_thread.daemon = True
                depth_thread.start()
                
                trade_thread = threading.Thread(target=self.trade_ws.run_forever)
                trade_thread.daemon = True
                trade_thread.start()
                
            else:
                # Mainnet uses combined stream format (like your working example)
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
                
                ws_thread = threading.Thread(target=self.ws.run_forever)
                ws_thread.daemon = True
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
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 50)
        logger.info("BINANCE SCALPER STARTING")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Quantity: {QTY}")
        logger.info(f"Min Spread: {MIN_SPREAD}")
        logger.info(f"Profit Target: {PROFIT_TARGET}")
        logger.info(f"Stop Loss: {STOP_LOSS}")
        logger.info(f"Testnet: {TESTNET}")
        logger.info("=" * 50)
        
        # Send startup notification
        startup_msg = f"🚀 Scalper Started!\n📊 {SYMBOL} | Qty: {QTY}\n📈 TP: {PROFIT_TARGET*100:.1f}% | SL: {STOP_LOSS*100:.1f}%"
        self.send_telegram(startup_msg)
        
        try:
            # Start WebSocket streams
            self.start_websocket_streams()
            
            # Main loop with periodic status updates
            last_status_time = time.time()
            while True:
                time.sleep(1)
                
                # Send periodic status update every 5 minutes
                current_time = time.time()
                if current_time - last_status_time > 300:  # 5 minutes
                    status_msg = f"🔄 Status: {'Trading' if self.trade_active else 'Monitoring'}"
                    if self.trade_active:
                        status_msg += f"\n💰 Entry: {self.entry_price:.6f}"
                        status_msg += f"\n📊 Current: {self.current_price:.6f}"
                    logger.info(status_msg.replace('\n', ' | '))
                    last_status_time = current_time
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.send_telegram(f"❌ Scalper error: {e}")
            
        finally:
            # Cleanup
            self.stop_websocket_streams()
            
            # If we have an active trade, send warning
            if self.trade_active:
                warning_msg = f"⚠️ Scalper stopped with active trade!\nEntry: {self.entry_price:.6f}"
                logger.warning(warning_msg.replace('\n', ' | '))
                self.send_telegram(warning_msg)
            
            logger.info("Scalper stopped")
            self.send_telegram("🛑 Scalper Stopped")

def main():
    """Main function"""
    # Validate required environment variables
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
#!/usr/bin/env python3
"""
AI Trading Analysis Agent - Complete Fixed Version
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

# For DeepSeek API
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_pct: float
    pnl_usd: float
    martingale_step: int
    reason: str
    spread_at_entry: float
    market_conditions: Dict

@dataclass
class ConfigUpdate:
    """Configuration update recommendation"""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    confidence: float

class TradingDatabase:
    """Handles trade data storage and retrieval"""
    
    def __init__(self, db_path: str = "/home/ubuntu/binance_scalper/trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
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
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    parameter TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    confidence REAL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def get_recent_trades(self, symbol: str, hours: int = 24) -> List[Trade]:
        """Get recent trades for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                logger.warning("Trades table does not exist yet")
                conn.close()
                return []
            
            df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours), conn, params=(symbol,))
            
            conn.close()
            
            if df.empty:
                logger.info(f"No trades found for {symbol} in last {hours} hours")
                return []
            
            trades = []
            for _, row in df.iterrows():
                try:
                    trades.append(Trade(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        symbol=row['symbol'],
                        side=row['side'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        quantity=row['quantity'],
                        pnl_pct=row['pnl_pct'],
                        pnl_usd=row['pnl_usd'],
                        martingale_step=row['martingale_step'],
                        reason=row['reason'],
                        spread_at_entry=row['spread_at_entry'],
                        market_conditions=json.loads(row['market_conditions'] or '{}')
                    ))
                except Exception as e:
                    logger.error(f"Error parsing trade row: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

class DeepSeekAnalyzer:
    """DeepSeek-powered trading analysis"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def analyze_trading_performance(self, trades: List[Trade], current_config: Dict) -> List[ConfigUpdate]:
        """Analyze trades and suggest configuration changes"""
        
        if not trades:
            logger.info("No trades to analyze")
            return []
        
        if len(trades) < 3:
            logger.info(f"Insufficient trades for analysis: {len(trades)} < 3")
            return []
        
        analysis_data = self._prepare_analysis_data(trades, current_config)
        prompt = self._create_analysis_prompt(analysis_data)
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trading analyst. Provide specific, actionable recommendations in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content
            logger.info("AI analysis completed")
            
            recommendations = self._parse_ai_recommendations(ai_response)
            return recommendations
            
        except Exception as e:
            logger.error(f"DeepSeek AI analysis failed: {e}")
            return []
    
    def _prepare_analysis_data(self, trades: List[Trade], config: Dict) -> Dict:
        """Prepare data for AI analysis"""
        
        df = pd.DataFrame([{
            'pnl_pct': t.pnl_pct,
            'reason': t.reason,
            'martingale_step': t.martingale_step,
            'spread_at_entry': t.spread_at_entry,
            'symbol': t.symbol
        } for t in trades])
        
        win_rate = len(df[df['pnl_pct'] > 0]) / len(df) if len(df) > 0 else 0
        avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if len(df[df['pnl_pct'] > 0]) > 0 else 0
        avg_loss = df[df['pnl_pct'] < 0]['pnl_pct'].mean() if len(df[df['pnl_pct'] < 0]) > 0 else 0
        total_pnl = df['pnl_pct'].sum()
        
        stop_loss_count = len(df[df['reason'] == 'stop_loss'])
        profit_target_count = len(df[df['reason'] == 'profit_target'])
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'stop_loss_count': stop_loss_count,
            'profit_target_count': profit_target_count,
            'current_config': config,
            'symbol': trades[0].symbol if trades else 'UNKNOWN'
        }
    
    def _create_analysis_prompt(self, data: Dict) -> str:
        """Create prompt for AI analysis"""
        
        return f"""
Analyze this cryptocurrency scalping bot performance and suggest specific configuration improvements:

PERFORMANCE METRICS:
- Symbol: {data['symbol']}
- Total Trades: {data['total_trades']}
- Win Rate: {data['win_rate']:.1%}
- Average Win: {data['avg_win']:.3%}
- Average Loss: {data['avg_loss']:.3%}
- Total P&L: {data['total_pnl']:.3%}
- Stop Loss Hits: {data['stop_loss_count']}
- Profit Targets Hit: {data['profit_target_count']}

CURRENT CONFIGURATION:
- Trade Quantity: {data['current_config'].get('TRADE_QUANTITY', 'N/A')}
- Min Spread: {data['current_config'].get('MIN_SPREAD', 'N/A')}
- Profit Target: {data['current_config'].get('PROFIT_TARGET', 'N/A')}
- Stop Loss: {data['current_config'].get('STOP_LOSS', 'N/A')}

Respond ONLY in this JSON format:
{{
    "analysis": "Brief analysis of main issues",
    "recommendations": [
        {{
            "parameter": "PROFIT_TARGET",
            "current_value": 0.002,
            "suggested_value": 0.003,
            "reason": "Specific reasoning",
            "confidence": 0.8
        }}
    ]
}}
"""
    
    def _parse_ai_recommendations(self, ai_response: str) -> List[ConfigUpdate]:
        """Parse AI response into configuration updates"""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in AI response")
                return []
            
            data = json.loads(json_match.group())
            recommendations = []
            
            for rec in data.get('recommendations', []):
                try:
                    recommendations.append(ConfigUpdate(
                        parameter=rec['parameter'],
                        old_value=float(rec['current_value']),
                        new_value=float(rec['suggested_value']),
                        reason=rec['reason'],
                        confidence=float(rec['confidence'])
                    ))
                except (KeyError, ValueError) as e:
                    logger.error(f"Error parsing recommendation: {e}")
                    continue
            
            logger.info(f"Parsed {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse AI recommendations: {e}")
            return []

class ConfigManager:
    """Manages configuration updates - COMPLETE VERSION"""
    
    def __init__(self, config_dir: str = "/home/ubuntu/binance_scalper/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, symbol: str) -> Dict:
        """Load current configuration"""
        config_file = self.config_dir / f"{symbol.lower()}.env"
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        config = {}
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return config
    
    def update_config(self, symbol: str, updates: List[ConfigUpdate]) -> bool:
        """Apply configuration updates - THE MISSING METHOD"""
        config_file = self.config_dir / f"{symbol.lower()}.env"
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return False
        
        try:
            # Read current config
            with open(config_file, 'r') as f:
                lines = f.readlines()
            
            # Create backup
            backup_file = config_file.with_suffix(f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            with open(backup_file, 'w') as f:
                f.writelines(lines)
            logger.info(f"Config backup created: {backup_file}")
            
            # Apply updates
            updated_count = 0
            for update in updates:
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{update.parameter}="):
                        old_line = lines[i].strip()
                        lines[i] = f"{update.parameter}={update.new_value}\n"
                        logger.info(f"Updated {update.parameter}: {update.old_value} -> {update.new_value}")
                        logger.info(f"Reason: {update.reason}")
                        updated_count += 1
                        break
            
            # Write updated config
            with open(config_file, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"Applied {updated_count} configuration updates to {symbol}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

class TradingAgent:
    """Main AI trading agent - COMPLETE VERSION"""
    
    def __init__(self, deepseek_api_key: str):
        self.db = TradingDatabase()
        self.analyzer = DeepSeekAnalyzer(deepseek_api_key)
        self.config_manager = ConfigManager()
    
    def analyze_and_update(self, symbol: str, min_trades: int = 3) -> bool:
        """Analyze recent performance and update configuration if needed"""
        
        logger.info(f"Starting analysis for {symbol}")
        
        # Get recent trades
        trades = self.db.get_recent_trades(symbol, hours=24)
        
        if len(trades) < min_trades:
            logger.info(f"Insufficient trades for analysis: {len(trades)} < {min_trades}")
            return False
        
        # Load current config
        current_config = self.config_manager.load_config(symbol)
        
        if not current_config:
            logger.error(f"Could not load configuration for {symbol}")
            return False
        
        # Get AI recommendations
        recommendations = self.analyzer.analyze_trading_performance(trades, current_config)
        
        if not recommendations:
            logger.info("No recommendations from AI analysis")
            return False
        
        # Filter high-confidence recommendations
        confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.75'))
        high_confidence = [r for r in recommendations if r.confidence >= confidence_threshold]
        
        if not high_confidence:
            logger.info(f"No high-confidence recommendations (threshold: {confidence_threshold})")
            return False
        
        # Apply updates
        success = self.config_manager.update_config(symbol, high_confidence)
        
        if success:
            logger.info(f"Applied {len(high_confidence)} configuration updates for {symbol}")
            
            # Log changes to database
            for rec in high_confidence:
                self._log_config_change(symbol, rec)
            
            # Restart service if enabled
            auto_restart = os.getenv('AUTO_RESTART_SERVICES', 'true').lower() == 'true'
            if auto_restart:
                logger.info(f"Restarting trading service for {symbol}")
                os.system(f"sudo systemctl restart binance-scalper@{symbol.lower()}")
        
        return success
    
    def _log_config_change(self, symbol: str, update: ConfigUpdate):
        """Log configuration change to database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO config_changes 
                (timestamp, symbol, parameter, old_value, new_value, reason, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                update.parameter,
                update.old_value,
                update.new_value,
                update.reason,
                update.confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log config change: {e}")

def main():
    """Main function"""
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY not found in environment")
        return
    
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,ARBUSDT,SHIBUSDT').split(',')
    analysis_interval = int(os.getenv('ANALYSIS_INTERVAL_HOURS', '2'))
    min_trades = int(os.getenv('MIN_TRADES_FOR_ANALYSIS', '3'))
    auto_update = os.getenv('AUTO_UPDATE_ENABLED', 'true').lower() == 'true'
    
    logger.info(f"AI Trading Agent started - Symbols: {symbols}, Interval: {analysis_interval}h, Auto-update: {auto_update}")
    
    agent = TradingAgent(deepseek_api_key)
    
    while True:
        try:
            for symbol in symbols:
                symbol = symbol.strip()
                logger.info(f"Analyzing {symbol}...")
                
                if auto_update:
                    success = agent.analyze_and_update(symbol, min_trades=min_trades)
                    if success:
                        logger.info(f"✅ Configuration updated for {symbol}")
                else:
                    trades = agent.db.get_recent_trades(symbol, hours=24)
                    logger.info(f"Analysis only mode: {len(trades)} trades found for {symbol}")
                
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
        
        logger.info(f"Waiting {analysis_interval} hours for next analysis cycle...")
        time.sleep(analysis_interval * 3600)

if __name__ == "__main__":
    main()

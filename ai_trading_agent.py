#!/usr/bin/env python3
"""
Enhanced AI Trading Analysis Agent - Complete Final Version
Version 2.0 - Statistical Validation & Robust Parsing
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import time

# For AI APIs
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
    """Configuration update recommendation with enhanced metadata"""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    confidence: float
    statistical_significance: float
    expected_impact: str
    risk_assessment: str

@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics"""
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    calmar_ratio: float
    statistical_significance: float

class TradingDatabase:
    """Enhanced database with performance tracking"""
    
    def __init__(self, db_path: str = "/home/ubuntu/binance_scalper/trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with enhanced tables"""
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
                    confidence REAL,
                    statistical_significance REAL,
                    model_used TEXT,
                    expected_impact TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Enhanced database initialized at {self.db_path}")
            
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
    """Enhanced DeepSeek-powered trading analysis"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def analyze_trading_performance(self, trades: List[Trade], current_config: Dict) -> List[ConfigUpdate]:
        """Enhanced analysis with statistical validation"""
        
        if not trades:
            logger.info("No trades to analyze")
            return []
        
        if len(trades) < 3:
            logger.info(f"Insufficient trades for analysis: {len(trades)} < 3")
            return []
        
        # Calculate enhanced metrics
        metrics = self._calculate_performance_metrics(trades)
        analysis_data = self._prepare_enhanced_analysis_data(trades, current_config, metrics)
        
        # Use enhanced prompting strategy
        prompt = self._create_enhanced_analysis_prompt(analysis_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            ai_response = response.choices[0].message.content
            logger.info("Enhanced AI analysis completed")
            
            # Log the raw AI response for debugging (first 500 chars)
            logger.debug(f"Raw AI response preview: {ai_response[:500]}...")
            
            recommendations = self._parse_enhanced_ai_recommendations(ai_response, metrics)
            validated_recommendations = self._validate_recommendations(recommendations, current_config)
            
            return validated_recommendations
            
        except Exception as e:
            logger.error(f"DeepSeek AI analysis failed: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """Enhanced system prompt for better AI responses"""
        return """You are an expert quantitative trading analyst specializing in cryptocurrency scalping strategies. 

Your expertise includes:
- Statistical analysis and hypothesis testing
- Risk management optimization
- Market microstructure analysis
- Algorithmic trading parameter tuning

Always provide:
1. Statistically significant recommendations (p < 0.05 when possible)
2. Risk-adjusted performance metrics
3. Confidence scores based on sample size and data quality
4. Clear reasoning for each recommendation

Focus on maximizing risk-adjusted returns while maintaining strict risk controls.
Respond ONLY in valid JSON format with decimal numbers (no % signs or $ symbols)."""
    
    def _calculate_performance_metrics(self, trades: List[Trade]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        df = pd.DataFrame([{
            'pnl_pct': t.pnl_pct,
            'pnl_usd': t.pnl_usd,
            'timestamp': t.timestamp
        } for t in trades])
        
        # Basic metrics
        win_rate = len(df[df['pnl_pct'] > 0]) / len(df) if len(df) > 0 else 0
        avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if len(df[df['pnl_pct'] > 0]) > 0 else 0
        avg_loss = df[df['pnl_pct'] < 0]['pnl_pct'].mean() if len(df[df['pnl_pct'] < 0]) > 0 else 0
        total_pnl = df['pnl_pct'].sum()
        
        # Advanced metrics
        returns = df['pnl_pct'].values
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        profit_factor = self._calculate_profit_factor(df)
        calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Statistical significance
        statistical_significance = min(len(trades) / 30, 1.0)  # Max confidence at 30+ trades
        
        return PerformanceMetrics(
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            statistical_significance=statistical_significance
        )
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        return mean_return / std_return if std_return != 0 else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
    def _calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        total_profits = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
        total_losses = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
        return total_profits / total_losses if total_losses != 0 else float('inf')
    
    def _prepare_enhanced_analysis_data(self, trades: List[Trade], config: Dict, metrics: PerformanceMetrics) -> Dict:
        """Prepare enhanced data for AI analysis"""
        
        df = pd.DataFrame([{
            'pnl_pct': t.pnl_pct,
            'reason': t.reason,
            'martingale_step': t.martingale_step,
            'spread_at_entry': t.spread_at_entry,
            'symbol': t.symbol,
            'timestamp': t.timestamp,
            'hour': t.timestamp.hour
        } for t in trades])
        
        # Market condition analysis
        stop_loss_count = len(df[df['reason'] == 'stop_loss'])
        profit_target_count = len(df[df['reason'] == 'profit_target'])
        
        # Time-based analysis
        hourly_performance = df.groupby('hour')['pnl_pct'].mean().to_dict()
        
        # Volatility analysis
        volatility = df['pnl_pct'].std()
        
        return {
            'metrics': metrics,
            'symbol': trades[0].symbol if trades else 'UNKNOWN',
            'stop_loss_count': stop_loss_count,
            'profit_target_count': profit_target_count,
            'hourly_performance': hourly_performance,
            'volatility': volatility,
            'current_config': config,
            'spread_analysis': df['spread_at_entry'].describe().to_dict()
        }
    
    def _create_enhanced_analysis_prompt(self, data: Dict) -> str:
        """Create enhanced prompt with statistical context"""
        
        metrics = data['metrics']
        
        return f"""
QUANTITATIVE TRADING ANALYSIS - CRYPTOCURRENCY SCALPING

STATISTICAL OVERVIEW:
Sample Size: {metrics.total_trades} trades (Significance: {'High' if metrics.statistical_significance > 0.8 else 'Moderate' if metrics.statistical_significance > 0.5 else 'Low'})
Symbol: {data['symbol']}

PERFORMANCE METRICS:
Win Rate: {metrics.win_rate:.1%} (Target: >55% for scalping)
Risk/Reward Ratio: {abs(metrics.avg_win/metrics.avg_loss) if metrics.avg_loss != 0 else 'N/A':.2f}
Sharpe Ratio: {metrics.sharpe_ratio:.2f} (Target: >1.0)
Maximum Drawdown: {metrics.max_drawdown:.2%} (Limit: <5%)
Profit Factor: {metrics.profit_factor:.2f} (Target: >1.3)
Calmar Ratio: {metrics.calmar_ratio:.2f}

CURRENT CONFIGURATION:
- Trade Quantity: {data['current_config'].get('TRADE_QUANTITY', 'N/A')}
- Min Spread: {data['current_config'].get('MIN_SPREAD', 'N/A')}
- Profit Target: {data['current_config'].get('PROFIT_TARGET', 'N/A')}
- Stop Loss: {data['current_config'].get('STOP_LOSS', 'N/A')}

TRADE OUTCOME ANALYSIS:
- Stop Losses Hit: {data['stop_loss_count']} ({data['stop_loss_count']/metrics.total_trades:.1%})
- Profit Targets Hit: {data['profit_target_count']} ({data['profit_target_count']/metrics.total_trades:.1%})

OPTIMIZATION GUIDELINES:
1. If Sharpe ratio < 1.0: Focus on risk reduction
2. If max drawdown > 3%: Implement stricter position sizing
3. If win rate < 50%: Reassess entry/exit criteria
4. If profit factor < 1.2: Optimize risk/reward ratio

STATISTICAL REQUIREMENTS:
- Only suggest changes with confidence >70%
- Consider sample size in confidence scoring
- Provide expected impact estimates

Respond with JSON format including statistical justification for each recommendation.

JSON Format Required:
{{
    "analysis_summary": "Brief statistical assessment",
    "primary_concerns": ["list", "of", "main", "issues"],
    "recommendations": [
        {{
            "parameter": "PARAMETER_NAME",
            "current_value": 0.0035,
            "suggested_value": 0.004,
            "reason": "Statistical justification with metrics",
            "confidence": 0.85,
            "expected_impact": "Specific expected outcome",
            "risk_assessment": "Low"
        }}
    ]
}}

CRITICAL: Use decimal numbers only (e.g., 0.0035 not 0.35% or $0.0035). No percentage signs, currency symbols, or text formatting in numeric values.
"""
    
    def _parse_enhanced_ai_recommendations(self, ai_response: str, metrics: PerformanceMetrics) -> List[ConfigUpdate]:
        """Parse AI response with enhanced validation and robust value parsing"""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in AI response")
                logger.error(f"AI response was: {ai_response[:500]}...")
                return []
            
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in AI response: {e}")
                logger.error(f"JSON content: {json_match.group()}")
                return []
            
            recommendations = []
            
            for i, rec in enumerate(data.get('recommendations', [])):
                try:
                    logger.debug(f"Processing recommendation {i+1}: {rec}")
                    
                    # Enhanced statistical significance calculation
                    base_confidence = self._parse_numeric_value(rec['confidence'])
                    if base_confidence is None:
                        logger.warning(f"Invalid confidence value in recommendation {i+1}: {rec.get('confidence')}")
                        continue
                    
                    statistical_significance = metrics.statistical_significance
                    
                    # Parse values with robust handling of formatting
                    old_value = self._parse_numeric_value(rec['current_value'])
                    new_value = self._parse_numeric_value(rec['suggested_value'])
                    
                    if old_value is None or new_value is None:
                        logger.warning(f"Skipping recommendation {i+1} due to unparseable values:")
                        logger.warning(f"  current_value: {rec.get('current_value')} -> {old_value}")
                        logger.warning(f"  suggested_value: {rec.get('suggested_value')} -> {new_value}")
                        continue
                    
                    # Validate parameter name
                    parameter = rec.get('parameter', '').upper()
                    if parameter not in ['PROFIT_TARGET', 'STOP_LOSS', 'MIN_SPREAD', 'TRADE_QUANTITY']:
                        logger.warning(f"Unknown parameter in recommendation {i+1}: {parameter}")
                        continue
                    
                    recommendations.append(ConfigUpdate(
                        parameter=parameter,
                        old_value=old_value,
                        new_value=new_value,
                        reason=rec.get('reason', 'No reason provided'),
                        confidence=min(base_confidence * statistical_significance, 1.0),  # Cap at 100%
                        statistical_significance=statistical_significance,
                        expected_impact=rec.get('expected_impact', 'Not specified'),
                        risk_assessment=rec.get('risk_assessment', 'Medium')
                    ))
                    
                    logger.info(f"✅ Parsed recommendation {i+1}: {parameter} {old_value} -> {new_value}")
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Error parsing recommendation {i+1}: {e}")
                    logger.error(f"Problematic recommendation data: {rec}")
                    continue
            
            logger.info(f"Successfully parsed {len(recommendations)} out of {len(data.get('recommendations', []))} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse AI recommendations: {e}")
            return []
    
    def _parse_numeric_value(self, value: Any) -> Optional[float]:
        """Robust numeric value parser that handles various formats"""
        if value is None:
            return None
            
        # Convert to string for processing
        str_value = str(value).strip()
        
        # Handle empty strings
        if not str_value:
            return None
        
        try:
            # Remove common formatting characters
            # Handle percentages (convert to decimal)
            if '%' in str_value:
                str_value = str_value.replace('%', '')
                numeric_value = float(str_value) / 100
                return numeric_value
            
            # Handle dollar signs and other currency symbols
            str_value = str_value.replace('$', '').replace('€', '').replace('£', '')
            
            # Handle commas in large numbers
            str_value = str_value.replace(',', '')
            
            # Handle scientific notation
            if 'e' in str_value.lower():
                return float(str_value)
            
            # Standard float conversion
            return float(str_value)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Could not parse numeric value '{value}': {e}")
            return None
    
    def _validate_recommendations(self, recommendations: List[ConfigUpdate], current_config: Dict) -> List[ConfigUpdate]:
        """Enhanced validation with safety bounds"""
        validated = []
        
        for rec in recommendations:
            # Safety bounds checking
            if rec.parameter == 'STOP_LOSS':
                if rec.new_value > 0.025:  # Max 2.5% stop loss
                    logger.warning(f"Stop loss too high: {rec.new_value:.4f}, capping at 0.025")
                    rec.new_value = 0.025
                    rec.confidence *= 0.7
                elif rec.new_value < 0.002:  # Min 0.2% stop loss
                    logger.warning(f"Stop loss too low: {rec.new_value:.4f}, setting to 0.002")
                    rec.new_value = 0.002
                    rec.confidence *= 0.7
            
            elif rec.parameter == 'PROFIT_TARGET':
                if rec.new_value > 0.015:  # Max 1.5% profit target
                    logger.warning(f"Profit target too high: {rec.new_value:.4f}")
                    rec.new_value = 0.015
                    rec.confidence *= 0.8
                elif rec.new_value < 0.0005:  # Min 0.05% profit target
                    logger.warning(f"Profit target too low: {rec.new_value:.4f}")
                    continue
            
            elif rec.parameter == 'MIN_SPREAD':
                if rec.new_value > 0.01:  # Max 1% spread requirement
                    logger.warning(f"Min spread too high: {rec.new_value:.4f}")
                    continue
            
            # Change magnitude validation
            if rec.old_value != 0:
                change_pct = abs(rec.new_value - rec.old_value) / rec.old_value
                if change_pct > 0.5:  # Max 50% change at once
                    logger.warning(f"Large parameter change: {change_pct:.1%} for {rec.parameter}")
                    rec.confidence *= 0.6
                    rec.risk_assessment = "High"
            
            # Confidence threshold
            min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
            if rec.confidence >= min_confidence:
                validated.append(rec)
            else:
                logger.info(f"Recommendation below confidence threshold: {rec.confidence:.2f} < {min_confidence}")
        
        return validated

class EnhancedConfigManager:
    """Enhanced configuration manager with versioning"""
    
    def __init__(self, config_dir: str = "/home/ubuntu/binance_scalper/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
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
                        try:
                            # Try to convert to appropriate type
                            if '.' in value:
                                config[key.strip()] = float(value.strip())
                            elif value.strip().isdigit():
                                config[key.strip()] = int(value.strip())
                            else:
                                config[key.strip()] = value.strip()
                        except ValueError:
                            config[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return config
    
    def update_config(self, symbol: str, updates: List[ConfigUpdate]) -> bool:
        """Apply configuration updates with enhanced logging"""
        config_file = self.config_dir / f"{symbol.lower()}.env"
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return False
        
        try:
            # Read current config
            with open(config_file, 'r') as f:
                lines = f.readlines()
            
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"{symbol.lower()}.env.backup.{timestamp}"
            with open(backup_file, 'w') as f:
                f.writelines(lines)
            logger.info(f"Config backup created: {backup_file}")
            
            # Apply updates with detailed logging
            updated_count = 0
            for update in updates:
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{update.parameter}="):
                        old_line = lines[i].strip()
                        lines[i] = f"{update.parameter}={update.new_value}\n"
                        
                        logger.info(f"✅ {update.parameter}: {update.old_value} → {update.new_value}")
                        logger.info(f"   Reason: {update.reason}")
                        logger.info(f"   Confidence: {update.confidence:.1%}")
                        logger.info(f"   Expected Impact: {update.expected_impact}")
                        logger.info(f"   Risk Assessment: {update.risk_assessment}")
                        
                        updated_count += 1
                        break
            
            # Write updated config with header
            header = f"# Updated by AI Agent on {datetime.now().isoformat()}\n"
            header += f"# {updated_count} parameters updated\n\n"
            
            with open(config_file, 'w') as f:
                f.write(header)
                f.writelines(lines)
            
            logger.info(f"Applied {updated_count} configuration updates to {symbol}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

class EnhancedTradingAgent:
    """Enhanced AI trading agent with improved analytics"""
    
    def __init__(self, deepseek_api_key: str):
        self.db = TradingDatabase()
        self.analyzer = DeepSeekAnalyzer(deepseek_api_key)
        self.config_manager = EnhancedConfigManager()
    
    def analyze_and_update(self, symbol: str, min_trades: int = 5) -> bool:
        """Enhanced analysis with performance tracking"""
        
        logger.info(f"🔍 Starting enhanced analysis for {symbol}")
        
        # Get recent trades
        trades = self.db.get_recent_trades(symbol, hours=24)
        
        if len(trades) < min_trades:
            logger.info(f"⏳ Insufficient trades for analysis: {len(trades)} < {min_trades}")
            return False
        
        # Load current config
        current_config = self.config_manager.load_config(symbol)
        
        if not current_config:
            logger.error(f"❌ Could not load configuration for {symbol}")
            return False
        
        # Get AI recommendations
        recommendations = self.analyzer.analyze_trading_performance(trades, current_config)
        
        if not recommendations:
            logger.info("💡 No recommendations from AI analysis")
            return False
        
        # Log recommendations summary
        logger.info(f"📊 Received {len(recommendations)} recommendations:")
        for rec in recommendations:
            logger.info(f"   • {rec.parameter}: {rec.old_value} → {rec.new_value} (confidence: {rec.confidence:.1%})")
        
        # Apply updates
        success = self.config_manager.update_config(symbol, recommendations)
        
        if success:
            logger.info(f"✅ Applied {len(recommendations)} configuration updates for {symbol}")
            
            # Log changes to database
            for rec in recommendations:
                self._log_enhanced_config_change(symbol, rec)
            
            # Restart service if enabled
            auto_restart = os.getenv('AUTO_RESTART_SERVICES', 'true').lower() == 'true'
            if auto_restart:
                logger.info(f"🔄 Restarting trading service for {symbol}")
                os.system(f"sudo systemctl restart binance-scalper@{symbol.lower()}")
        
        return success
    
    def _log_enhanced_config_change(self, symbol: str, update: ConfigUpdate):
        """Log enhanced configuration change to database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO config_changes 
                (timestamp, symbol, parameter, old_value, new_value, reason, confidence, 
                 statistical_significance, model_used, expected_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                update.parameter,
                update.old_value,
                update.new_value,
                update.reason,
                update.confidence,
                update.statistical_significance,
                'deepseek-chat',
                update.expected_impact
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log config change: {e}")

def main():
    """Enhanced main function with better error handling"""
    
    # Environment validation
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        logger.error("❌ DEEPSEEK_API_KEY not found in environment")
        return
    
    # Configuration
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,ARBUSDT,SHIBUSDT').split(',')
    analysis_interval = int(os.getenv('ANALYSIS_INTERVAL_HOURS', '4'))
    min_trades = int(os.getenv('MIN_TRADES_FOR_ANALYSIS', '5'))
    auto_update = os.getenv('AUTO_UPDATE_ENABLED', 'true').lower() == 'true'
    
    logger.info(f"""
🚀 Enhanced AI Trading Agent Started
   📈 Symbols: {symbols}
   ⏱️  Analysis Interval: {analysis_interval}h
   📊 Min Trades Required: {min_trades}
   🔧 Auto-update: {auto_update}
   🤖 AI Model: DeepSeek Chat
""")
    
    agent = EnhancedTradingAgent(deepseek_api_key)
    
    while True:
        try:
            for symbol in symbols:
                symbol = symbol.strip()
                logger.info(f"🔍 Analyzing {symbol}...")
                
                if auto_update:
                    success = agent.analyze_and_update(symbol, min_trades=min_trades)
                    if success:
                        logger.info(f"✅ Configuration optimized for {symbol}")
                    else:
                        logger.info(f"⏸️  No changes needed for {symbol}")
                else:
                    trades = agent.db.get_recent_trades(symbol, hours=24)
                    logger.info(f"📊 Analysis-only mode: {len(trades)} trades found for {symbol}")
                
                time.sleep(30)  # Pause between symbols
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down AI Trading Agent...")
            break
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {e}")
        
        logger.info(f"⏳ Waiting {analysis_interval} hours for next analysis cycle...")
        time.sleep(analysis_interval * 3600)

if __name__ == "__main__":
    main()
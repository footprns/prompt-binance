#!/usr/bin/env python3
"""
Enhanced AI Trading Analysis Agent - Extended Parameter Tuning
Version 2.1 - Added Timing and Martingale Parameter Optimization
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
    # Extended metrics for timing analysis
    avg_trade_frequency: float
    martingale_usage_rate: float
    consecutive_loss_streaks: int

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
    """Enhanced DeepSeek-powered trading analysis with extended parameters"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def analyze_trading_performance(self, trades: List[Trade], current_config: Dict) -> List[ConfigUpdate]:
        """Enhanced analysis with timing and martingale parameter optimization"""
        
        if not trades:
            logger.info("No trades to analyze")
            return []
        
        if len(trades) < 3:
            logger.info(f"Insufficient trades for analysis: {len(trades)} < 3")
            return []
        
        # Calculate enhanced metrics including timing analysis
        metrics = self._calculate_enhanced_performance_metrics(trades)
        analysis_data = self._prepare_extended_analysis_data(trades, current_config, metrics)
        
        # Use enhanced prompting strategy with extended parameters
        prompt = self._create_extended_analysis_prompt(analysis_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_extended_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000  # Increased for more comprehensive analysis
            )
            
            ai_response = response.choices[0].message.content
            logger.info("Extended AI analysis completed")
            
            # Log the raw AI response for debugging (first 500 chars)
            logger.debug(f"Raw AI response preview: {ai_response[:500]}...")
            
            recommendations = self._parse_extended_ai_recommendations(ai_response, metrics)
            validated_recommendations = self._validate_extended_recommendations(recommendations, current_config)
            
            return validated_recommendations
            
        except Exception as e:
            logger.error(f"DeepSeek AI analysis failed: {e}")
            return []
    
    def _get_extended_system_prompt(self) -> str:
        """Extended system prompt for comprehensive parameter optimization"""
        return """You are an expert quantitative trading analyst specializing in cryptocurrency scalping strategies with deep expertise in:

- Statistical analysis and hypothesis testing
- Risk management optimization including position sizing and drawdown control
- Market microstructure analysis and timing optimization
- Martingale system optimization and risk control
- Trading frequency analysis and cooldown strategies
- Algorithmic trading parameter tuning across all dimensions

Your analysis scope includes:
1. Core trading parameters (profit targets, stop losses, spreads)
2. Timing and frequency controls (cooldowns, trade intervals)
3. Martingale and position scaling strategies
4. Risk management systems

Always provide:
1. Statistically significant recommendations (p < 0.05 when possible)
2. Risk-adjusted performance metrics
3. Confidence scores based on sample size and data quality
4. Clear reasoning for each recommendation
5. Safety-first approach to parameter changes

Focus on maximizing risk-adjusted returns while maintaining strict risk controls.
Respond ONLY in valid JSON format with decimal numbers (no % signs or $ symbols)."""
    
    def _calculate_enhanced_performance_metrics(self, trades: List[Trade]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics including timing analysis"""
        df = pd.DataFrame([{
            'pnl_pct': t.pnl_pct,
            'pnl_usd': t.pnl_usd,
            'timestamp': t.timestamp,
            'martingale_step': t.martingale_step
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
        
        # Extended metrics for timing analysis
        avg_trade_frequency = self._calculate_trade_frequency(df)
        martingale_usage_rate = len(df[df['martingale_step'] > 0]) / len(df) if len(df) > 0 else 0
        consecutive_loss_streaks = self._calculate_max_consecutive_losses(df)
        
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
            statistical_significance=statistical_significance,
            avg_trade_frequency=avg_trade_frequency,
            martingale_usage_rate=martingale_usage_rate,
            consecutive_loss_streaks=consecutive_loss_streaks
        )
    
    def _calculate_trade_frequency(self, df: pd.DataFrame) -> float:
        """Calculate average trades per hour"""
        if len(df) < 2:
            return 0
        
        df_sorted = df.sort_values('timestamp')
        time_span_hours = (df_sorted.iloc[-1]['timestamp'] - df_sorted.iloc[0]['timestamp']).total_seconds() / 3600
        return len(df) / time_span_hours if time_span_hours > 0 else 0
    
    def _calculate_max_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losing trades"""
        if len(df) == 0:
            return 0
        
        df_sorted = df.sort_values('timestamp')
        max_streak = 0
        current_streak = 0
        
        for _, row in df_sorted.iterrows():
            if row['pnl_pct'] < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
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
    
    def _prepare_extended_analysis_data(self, trades: List[Trade], config: Dict, metrics: PerformanceMetrics) -> Dict:
        """Prepare extended data for AI analysis including timing metrics"""
        
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
        
        # Extended timing analysis
        trade_intervals = self._calculate_trade_intervals(df)
        
        return {
            'metrics': metrics,
            'symbol': trades[0].symbol if trades else 'UNKNOWN',
            'stop_loss_count': stop_loss_count,
            'profit_target_count': profit_target_count,
            'hourly_performance': hourly_performance,
            'volatility': volatility,
            'current_config': config,
            'spread_analysis': df['spread_at_entry'].describe().to_dict(),
            'trade_intervals': trade_intervals,
            'martingale_analysis': df['martingale_step'].describe().to_dict()
        }
    
    def _calculate_trade_intervals(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics on time intervals between trades"""
        if len(df) < 2:
            return {'mean_interval_minutes': 0, 'min_interval_minutes': 0, 'max_interval_minutes': 0}
        
        df_sorted = df.sort_values('timestamp')
        intervals = []
        
        for i in range(1, len(df_sorted)):
            interval = (df_sorted.iloc[i]['timestamp'] - df_sorted.iloc[i-1]['timestamp']).total_seconds() / 60
            intervals.append(interval)
        
        if intervals:
            return {
                'mean_interval_minutes': np.mean(intervals),
                'min_interval_minutes': np.min(intervals),
                'max_interval_minutes': np.max(intervals)
            }
        
        return {'mean_interval_minutes': 0, 'min_interval_minutes': 0, 'max_interval_minutes': 0}
    
    def _create_extended_analysis_prompt(self, data: Dict) -> str:
        """Create comprehensive prompt including timing and martingale analysis"""
        
        metrics = data['metrics']
        
        return f"""
COMPREHENSIVE QUANTITATIVE TRADING ANALYSIS - CRYPTOCURRENCY SCALPING

STATISTICAL OVERVIEW:
Sample Size: {metrics.total_trades} trades (Significance: {'High' if metrics.statistical_significance > 0.8 else 'Moderate' if metrics.statistical_significance > 0.5 else 'Low'})
Symbol: {data['symbol']}

CORE PERFORMANCE METRICS:
Win Rate: {metrics.win_rate:.1%} (Target: >55% for scalping)
Risk/Reward Ratio: {abs(metrics.avg_win/metrics.avg_loss) if metrics.avg_loss != 0 else 'N/A':.2f}
Sharpe Ratio: {metrics.sharpe_ratio:.2f} (Target: >1.0)
Maximum Drawdown: {metrics.max_drawdown:.2%} (Limit: <5%)
Profit Factor: {metrics.profit_factor:.2f} (Target: >1.3)
Total PnL: {metrics.total_pnl:.2%}

TIMING & FREQUENCY ANALYSIS:
Average Trade Frequency: {metrics.avg_trade_frequency:.2f} trades/hour
Mean Interval Between Trades: {data['trade_intervals']['mean_interval_minutes']:.1f} minutes
Consecutive Loss Streaks: {metrics.consecutive_loss_streaks} (Max recommended: 3)

MARTINGALE ANALYSIS:
Martingale Usage Rate: {metrics.martingale_usage_rate:.1%}
Martingale Step Distribution: {data['martingale_analysis']}

CURRENT CONFIGURATION:
CORE TRADING:
- Trade Quantity: {data['current_config'].get('TRADE_QUANTITY', 'N/A')}
- Min Spread: {data['current_config'].get('MIN_SPREAD', 'N/A')}
- Profit Target: {data['current_config'].get('PROFIT_TARGET', 'N/A')}
- Stop Loss: {data['current_config'].get('STOP_LOSS', 'N/A')}

TIMING CONTROLS:
- Cooldown Seconds: {data['current_config'].get('COOLDOWN_SECONDS', 'N/A')}
- Extended Cooldown: {data['current_config'].get('EXTENDED_COOLDOWN_SECONDS', 'N/A')}

MARTINGALE SETTINGS:
- Martingale Enabled: {data['current_config'].get('MARTINGALE_ENABLED', 'N/A')}
- Martingale Multiplier: {data['current_config'].get('MARTINGALE_MULTIPLIER', 'N/A')}
- Martingale Max Steps: {data['current_config'].get('MARTINGALE_MAX_STEPS', 'N/A')}

TRADE OUTCOME ANALYSIS:
- Stop Losses Hit: {data['stop_loss_count']} ({data['stop_loss_count']/metrics.total_trades:.1%})
- Profit Targets Hit: {data['profit_target_count']} ({data['profit_target_count']/metrics.total_trades:.1%})

OPTIMIZATION GUIDELINES:
1. CORE TRADING: If Sharpe ratio < 1.0 or profit factor < 1.2: Adjust targets/stops
2. TIMING: If trade frequency too high (>10/hour): Increase cooldowns
3. MARTINGALE: If consecutive losses > 3: Reduce max steps or multiplier
4. RISK: If max drawdown > 3%: Implement stricter controls

PARAMETER OPTIMIZATION SCOPE:
Core Trading: PROFIT_TARGET, STOP_LOSS, MIN_SPREAD, TRADE_QUANTITY
Timing Controls: COOLDOWN_SECONDS, EXTENDED_COOLDOWN_SECONDS  
Martingale System: MARTINGALE_MULTIPLIER, MARTINGALE_MAX_STEPS

STATISTICAL REQUIREMENTS:
- Only suggest changes with confidence >70%
- Consider sample size in confidence scoring
- Provide expected impact estimates
- Prioritize risk reduction over profit optimization

JSON Format Required:
{{
    "analysis_summary": "Brief statistical assessment including timing and martingale analysis",
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

CRITICAL: Use decimal numbers only (e.g., 0.0035 not 0.35% or $0.0035). For timing parameters use seconds (e.g., 300 for 5 minutes). No percentage signs, currency symbols, or text formatting in numeric values.
"""
    
    def _parse_extended_ai_recommendations(self, ai_response: str, metrics: PerformanceMetrics) -> List[ConfigUpdate]:
        """Parse AI response with extended parameter validation"""
        
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
                    
                    # Extended parameter validation
                    parameter = rec.get('parameter', '').upper()
                    valid_parameters = [
                        'PROFIT_TARGET', 'STOP_LOSS', 'MIN_SPREAD', 'TRADE_QUANTITY',
                        'COOLDOWN_SECONDS', 'EXTENDED_COOLDOWN_SECONDS',
                        'MARTINGALE_MULTIPLIER', 'MARTINGALE_MAX_STEPS'
                    ]
                    
                    if parameter not in valid_parameters:
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
    
    def _validate_extended_recommendations(self, recommendations: List[ConfigUpdate], current_config: Dict) -> List[ConfigUpdate]:
        """Enhanced validation with safety bounds for all parameters"""
        validated = []
        
        for rec in recommendations:
            # Core trading parameter validation
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
                elif rec.new_value < 0.001:  # Min 0.1% spread
                    logger.warning(f"Min spread too low: {rec.new_value:.4f}")
                    rec.new_value = 0.001
                    rec.confidence *= 0.8
            
            elif rec.parameter == 'TRADE_QUANTITY':
                if rec.new_value > 1000:  # Max quantity limit
                    logger.warning(f"Trade quantity too high: {rec.new_value}")
                    rec.new_value = 1000
                    rec.confidence *= 0.7
                elif rec.new_value < 10:  # Min quantity limit
                    logger.warning(f"Trade quantity too low: {rec.new_value}")
                    rec.new_value = 10
                    rec.confidence *= 0.8
            
            # Timing parameter validation
            elif rec.parameter == 'COOLDOWN_SECONDS':
                if rec.new_value > 3600:  # Max 1 hour cooldown
                    logger.warning(f"Cooldown too long: {rec.new_value}s, capping at 3600s")
                    rec.new_value = 3600
                    rec.confidence *= 0.8
                elif rec.new_value < 30:  # Min 30 seconds cooldown
                    logger.warning(f"Cooldown too short: {rec.new_value}s, setting to 30s")
                    rec.new_value = 30
                    rec.confidence *= 0.7
            
            elif rec.parameter == 'EXTENDED_COOLDOWN_SECONDS':
                if rec.new_value > 14400:  # Max 4 hours extended cooldown
                    logger.warning(f"Extended cooldown too long: {rec.new_value}s")
                    rec.new_value = 14400
                    rec.confidence *= 0.8
                elif rec.new_value < 300:  # Min 5 minutes extended cooldown
                    logger.warning(f"Extended cooldown too short: {rec.new_value}s")
                    rec.new_value = 300
                    rec.confidence *= 0.7
            
            # Martingale parameter validation
            elif rec.parameter == 'MARTINGALE_MULTIPLIER':
                if rec.new_value > 3.0:  # Max 3x multiplier for safety
                    logger.warning(f"Martingale multiplier too high: {rec.new_value}, capping at 3.0")
                    rec.new_value = 3.0
                    rec.confidence *= 0.6
                elif rec.new_value < 1.0:  # Min 1.0 (no scaling down)
                    logger.warning(f"Martingale multiplier too low: {rec.new_value}")
                    rec.new_value = 1.0
                    rec.confidence *= 0.8
            
            elif rec.parameter == 'MARTINGALE_MAX_STEPS':
                if rec.new_value > 5:  # Max 5 steps to prevent huge losses
                    logger.warning(f"Martingale max steps too high: {rec.new_value}, capping at 5")
                    rec.new_value = 5
                    rec.confidence *= 0.6
                    rec.risk_assessment = "High"
                elif rec.new_value < 0:  # Min 0 (disable martingale)
                    logger.warning(f"Martingale max steps negative: {rec.new_value}, setting to 0")
                    rec.new_value = 0
                    rec.confidence *= 0.9
            
            # Change magnitude validation
            if rec.old_value != 0:
                change_pct = abs(rec.new_value - rec.old_value) / rec.old_value
                if change_pct > 0.5:  # Max 50% change at once
                    logger.warning(f"Large parameter change: {change_pct:.1%} for {rec.parameter}")
                    rec.confidence *= 0.6
                    rec.risk_assessment = "High"
            
            # Special risk assessment for critical parameters
            if rec.parameter in ['MARTINGALE_MULTIPLIER', 'MARTINGALE_MAX_STEPS']:
                if rec.risk_assessment == "Low":
                    rec.risk_assessment = "Medium"  # Martingale is inherently risky
            
            # Confidence threshold with parameter-specific minimums
            min_confidence_base = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
            
            # Higher confidence required for risky parameters
            if rec.parameter in ['MARTINGALE_MULTIPLIER', 'MARTINGALE_MAX_STEPS']:
                min_confidence = max(min_confidence_base, 0.75)
            elif rec.parameter in ['COOLDOWN_SECONDS', 'EXTENDED_COOLDOWN_SECONDS']:
                min_confidence = max(min_confidence_base, 0.65)
            else:
                min_confidence = min_confidence_base
            
            if rec.confidence >= min_confidence:
                validated.append(rec)
                logger.info(f"✅ Validated recommendation: {rec.parameter} (confidence: {rec.confidence:.2f})")
            else:
                logger.info(f"❌ Recommendation below confidence threshold: {rec.parameter} ({rec.confidence:.2f} < {min_confidence})")
        
        return validated

class EnhancedConfigManager:
    """Enhanced configuration manager with versioning and extended parameter support"""
    
    def __init__(self, config_dir: str = "/home/ubuntu/binance_scalper/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def load_config(self, symbol: str) -> Dict:
        """Load current configuration including extended parameters"""
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
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle boolean values
                        if value.lower() in ['true', 'false']:
                            config[key] = value.lower() == 'true'
                        else:
                            try:
                                # Try to convert to appropriate numeric type
                                if '.' in value:
                                    config[key] = float(value)
                                elif value.isdigit():
                                    config[key] = int(value)
                                else:
                                    config[key] = value
                            except ValueError:
                                config[key] = value
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return config
    
    def update_config(self, symbol: str, updates: List[ConfigUpdate]) -> bool:
        """Apply configuration updates with enhanced logging and validation"""
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
            logger.info(f"📄 Config backup created: {backup_file}")
            
            # Apply updates with detailed logging
            updated_count = 0
            for update in updates:
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{update.parameter}="):
                        old_line = lines[i].strip()
                        
                        # Format value based on parameter type
                        if update.parameter in ['MARTINGALE_MAX_STEPS', 'COOLDOWN_SECONDS', 'EXTENDED_COOLDOWN_SECONDS', 'TRADE_QUANTITY']:
                            # Integer parameters
                            formatted_value = str(int(update.new_value))
                        else:
                            # Float parameters
                            formatted_value = str(update.new_value)
                        
                        lines[i] = f"{update.parameter}={formatted_value}\n"
                        
                        logger.info(f"✅ {update.parameter}: {update.old_value} → {update.new_value}")
                        logger.info(f"   📊 Reason: {update.reason}")
                        logger.info(f"   🎯 Confidence: {update.confidence:.1%}")
                        logger.info(f"   📈 Expected Impact: {update.expected_impact}")
                        logger.info(f"   ⚠️  Risk Assessment: {update.risk_assessment}")
                        
                        updated_count += 1
                        break
                else:
                    # Parameter not found, add it at the end
                    if update.parameter in ['MARTINGALE_MAX_STEPS', 'COOLDOWN_SECONDS', 'EXTENDED_COOLDOWN_SECONDS', 'TRADE_QUANTITY']:
                        formatted_value = str(int(update.new_value))
                    else:
                        formatted_value = str(update.new_value)
                    
                    lines.append(f"{update.parameter}={formatted_value}\n")
                    logger.info(f"➕ Added new parameter: {update.parameter}={formatted_value}")
                    updated_count += 1
            
            # Write updated config with enhanced header
            header = f"# Updated by Enhanced AI Agent on {datetime.now().isoformat()}\n"
            header += f"# {updated_count} parameters updated\n"
            header += f"# Extended parameter tuning: Core Trading + Timing + Martingale\n\n"
            
            with open(config_file, 'w') as f:
                f.write(header)
                f.writelines(lines)
            
            logger.info(f"🎉 Applied {updated_count} configuration updates to {symbol}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to update config: {e}")
            return False

class EnhancedTradingAgent:
    """Enhanced AI trading agent with comprehensive parameter optimization"""
    
    def __init__(self, deepseek_api_key: str):
        self.db = TradingDatabase()
        self.analyzer = DeepSeekAnalyzer(deepseek_api_key)
        self.config_manager = EnhancedConfigManager()
    
    def analyze_and_update(self, symbol: str, min_trades: int = 5) -> bool:
        """Enhanced analysis with comprehensive parameter optimization"""
        
        logger.info(f"🔍 Starting comprehensive analysis for {symbol}")
        
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
        
        # Get AI recommendations with extended parameter analysis
        recommendations = self.analyzer.analyze_trading_performance(trades, current_config)
        
        if not recommendations:
            logger.info("💡 No recommendations from comprehensive AI analysis")
            return False
        
        # Categorize recommendations for better logging
        core_params = ['PROFIT_TARGET', 'STOP_LOSS', 'MIN_SPREAD', 'TRADE_QUANTITY']
        timing_params = ['COOLDOWN_SECONDS', 'EXTENDED_COOLDOWN_SECONDS']
        martingale_params = ['MARTINGALE_MULTIPLIER', 'MARTINGALE_MAX_STEPS']
        
        core_recs = [r for r in recommendations if r.parameter in core_params]
        timing_recs = [r for r in recommendations if r.parameter in timing_params]
        martingale_recs = [r for r in recommendations if r.parameter in martingale_params]
        
        # Log recommendations summary by category
        logger.info(f"📊 Received {len(recommendations)} total recommendations:")
        if core_recs:
            logger.info(f"   🎯 Core Trading ({len(core_recs)}): {[r.parameter for r in core_recs]}")
        if timing_recs:
            logger.info(f"   ⏰ Timing Controls ({len(timing_recs)}): {[r.parameter for r in timing_recs]}")
        if martingale_recs:
            logger.info(f"   📈 Martingale System ({len(martingale_recs)}): {[r.parameter for r in martingale_recs]}")
        
        # Apply updates
        success = self.config_manager.update_config(symbol, recommendations)
        
        if success:
            logger.info(f"✅ Applied comprehensive optimization for {symbol}")
            
            # Log changes to database with enhanced metadata
            for rec in recommendations:
                self._log_enhanced_config_change(symbol, rec)
            
            # Restart service if enabled
            auto_restart = os.getenv('AUTO_RESTART_SERVICES', 'true').lower() == 'true'
            if auto_restart:
                logger.info(f"🔄 Restarting trading service for {symbol}")
                os.system(f"sudo systemctl restart binance-scalper@{symbol.lower()}")
            
            # Log summary of changes by category
            if core_recs:
                logger.info(f"🎯 Core trading parameters optimized: Risk-reward rebalanced")
            if timing_recs:
                logger.info(f"⏰ Timing controls adjusted: Trade frequency optimized")
            if martingale_recs:
                logger.info(f"📈 Martingale system tuned: Risk scaling controlled")
        
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
                'deepseek-chat-extended',
                update.expected_impact
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log config change: {e}")

def get_available_symbols(config_dir: str = "/home/ubuntu/binance_scalper/configs") -> List[str]:
    """Dynamically discover available symbols from configs directory"""
    try:
        config_path = Path(config_dir)
        if not config_path.exists():
            logger.warning(f"Config directory not found: {config_dir}")
            return []
        
        symbols = []
        for config_file in config_path.glob("*.env"):
            if config_file.name != ".env_template":  # Skip template file
                symbol = config_file.stem.upper()  # Convert btcusdt.env -> BTCUSDT
                symbols.append(symbol)
        
        symbols.sort()  # Sort alphabetically for consistent ordering
        return symbols
        
    except Exception as e:
        logger.error(f"Error discovering symbols from configs: {e}")
        return []

def validate_symbol_configs(symbols: List[str], config_dir: str = "/home/ubuntu/binance_scalper/configs") -> List[str]:
    """Validate that symbol config files exist and are readable"""
    valid_symbols = []
    
    for symbol in symbols:
        config_file = Path(config_dir) / f"{symbol.lower()}.env"
        
        if not config_file.exists():
            logger.warning(f"⚠️  Config file missing for {symbol}: {config_file}")
            continue
            
        try:
            # Test if we can read the config
            with open(config_file, 'r') as f:
                content = f.read()
                if 'TRADE_SYMBOL' in content or 'PROFIT_TARGET' in content:
                    valid_symbols.append(symbol)
                    logger.debug(f"✅ Valid config found for {symbol}")
                else:
                    logger.warning(f"⚠️  Config for {symbol} appears incomplete")
                    
        except Exception as e:
            logger.error(f"❌ Error reading config for {symbol}: {e}")
            continue
    
    return valid_symbols

def main():
    """Enhanced main function with dynamic symbol discovery"""
    
    # Environment validation
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        logger.error("❌ DEEPSEEK_API_KEY not found in environment")
        return
    
    # Configuration
    config_dir = os.getenv('CONFIG_DIR', '/home/ubuntu/binance_scalper/configs')
    analysis_interval = int(os.getenv('ANALYSIS_INTERVAL_HOURS', '4'))
    min_trades = int(os.getenv('MIN_TRADES_FOR_ANALYSIS', '5'))
    auto_update = os.getenv('AUTO_UPDATE_ENABLED', 'true').lower() == 'true'
    
    # Dynamic symbol discovery
    logger.info(f"🔍 Discovering symbols from config directory: {config_dir}")
    
    # First, try to get symbols from configs directory
    discovered_symbols = get_available_symbols(config_dir)
    
    if discovered_symbols:
        # Validate the discovered configs
        symbols = validate_symbol_configs(discovered_symbols, config_dir)
        logger.info(f"📂 Discovered {len(discovered_symbols)} config files")
        logger.info(f"✅ Validated {len(symbols)} usable configs")
    else:
        # Fallback to environment variable or default dashboard symbols
        env_symbols = os.getenv('SYMBOLS', '')
        if env_symbols:
            symbols = [s.strip().upper() for s in env_symbols.split(',')]
            logger.info(f"📝 Using symbols from environment: {symbols}")
        else:
            # Use the same symbols as dashboard.py
            symbols = ['BTCUSDT', 'ETHUSDT', 'ARBUSDT', 'SHIBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT']
            logger.info(f"🎯 Using default dashboard symbols: {symbols}")
    
    if not symbols:
        logger.error("❌ No valid symbols found. Please check your config directory or set SYMBOLS environment variable.")
        return
    
    logger.info(f"""
🚀 Enhanced AI Trading Agent Started - Dynamic Symbol Discovery
   📂 Config Directory: {config_dir}
   📈 Active Symbols ({len(symbols)}): {symbols}
   ⏱️  Analysis Interval: {analysis_interval}h
   📊 Min Trades Required: {min_trades}
   🔧 Auto-update: {auto_update}
   🤖 AI Model: DeepSeek Chat Extended
   
   🎯 Optimization Scope:
      • Core Trading: Profit/Loss targets, Spreads, Position sizing
      • Timing Controls: Cooldowns, Trade frequency management
      • Martingale System: Risk scaling and step controls
      
   💡 Symbol Discovery: 
      • Automatic detection from configs/ folder
      • Validates config file readability
      • Matches dashboard.py symbol list
""")
    
    agent = EnhancedTradingAgent(deepseek_api_key)
    
    # Initial symbol validation check
    logger.info("🔍 Performing initial symbol validation...")
    for symbol in symbols[:3]:  # Check first 3 symbols as examples
        config = agent.config_manager.load_config(symbol)
        if config:
            key_params = [k for k in config.keys() if k in ['PROFIT_TARGET', 'STOP_LOSS', 'TRADE_QUANTITY']]
            logger.info(f"   📋 {symbol}: {len(key_params)} core parameters found")
        else:
            logger.warning(f"   ⚠️  {symbol}: No config data loaded")
    
    while True:
        try:
            # Re-discover symbols periodically in case new configs are added
            current_symbols = get_available_symbols(config_dir)
            if current_symbols and len(current_symbols) != len(symbols):
                logger.info(f"🔄 Symbol list updated: {len(symbols)} -> {len(current_symbols)}")
                symbols = validate_symbol_configs(current_symbols, config_dir)
                logger.info(f"📈 New active symbols: {symbols}")
            
            for symbol in symbols:
                symbol = symbol.strip().upper()
                logger.info(f"🔍 Comprehensive analysis for {symbol}...")
                
                # Check if config file still exists before analysis
                config_file = Path(config_dir) / f"{symbol.lower()}.env"
                if not config_file.exists():
                    logger.warning(f"⚠️  Config file missing for {symbol}, skipping...")
                    continue
                
                if auto_update:
                    success = agent.analyze_and_update(symbol, min_trades=min_trades)
                    if success:
                        logger.info(f"✅ Full parameter optimization completed for {symbol}")
                    else:
                        logger.info(f"⏸️  No optimization needed for {symbol}")
                else:
                    trades = agent.db.get_recent_trades(symbol, hours=24)
                    logger.info(f"📊 Analysis-only mode: {len(trades)} trades found for {symbol}")
                
                time.sleep(30)  # Pause between symbols
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Enhanced AI Trading Agent...")
            break
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {e}")
        
        logger.info(f"⏳ Waiting {analysis_interval} hours for next comprehensive analysis cycle...")
        time.sleep(analysis_interval * 3600)

if __name__ == "__main__":
    main()
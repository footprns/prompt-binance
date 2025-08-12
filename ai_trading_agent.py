#!/usr/bin/env python3
"""
Complete Enhanced AI Trading Analysis Agent - Win Rate Optimization Focus
Version 3.0 - Complete implementation with all features
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
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
class EnhancedTrade:
    """Enhanced trade data structure with win rate analytics"""
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
    # New fields for win rate analysis
    confidence_score: float = 0.0
    market_regime: str = "NORMAL"
    entry_hour: int = 0
    trailing_stop_used: bool = False

@dataclass
class WinRateMetrics:
    """Win rate focused performance metrics"""
    total_trades: int
    win_rate: float
    loss_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    profit_factor: float
    # Enhanced win rate analytics
    win_rate_by_hour: Dict[int, float]
    win_rate_by_confidence: Dict[str, float]
    win_rate_by_regime: Dict[str, float]
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_losses: int
    trailing_stop_effectiveness: float

@dataclass
class WinRateConfigUpdate:
    """Configuration update focused on win rate improvement"""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    confidence: float
    expected_win_rate_impact: str
    risk_level: str
    priority: str  # HIGH, MEDIUM, LOW

class EnhancedTradingDatabase:
    """Enhanced database with win rate analytics"""
    
    def __init__(self, db_path: str = "/home/ubuntu/binance_scalper/trading_data.db"):
        self.db_path = db_path
        self.init_enhanced_database()
    
    def init_enhanced_database(self):
        """Initialize database with enhanced win rate tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced trades table
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
                    market_conditions TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    market_regime TEXT DEFAULT 'NORMAL',
                    entry_hour INTEGER DEFAULT 0,
                    trailing_stop_used INTEGER DEFAULT 0
                )
            ''')
            
            # Win rate analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS win_rate_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    period_hours INTEGER,
                    total_trades INTEGER,
                    win_rate REAL,
                    avg_win_pct REAL,
                    avg_loss_pct REAL,
                    profit_factor REAL,
                    max_consecutive_losses INTEGER,
                    best_hour INTEGER,
                    worst_hour INTEGER,
                    recommendations TEXT
                )
            ''')
            
            # Config changes table
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
                    expected_impact TEXT,
                    priority TEXT DEFAULT 'MEDIUM',
                    win_rate_focused INTEGER DEFAULT 0
                )
            ''')
            
            # Add new columns to existing tables if they don't exist
            existing_columns = self._get_table_columns('trades')
            new_columns = [
                ('confidence_score', 'REAL DEFAULT 0.0'),
                ('market_regime', 'TEXT DEFAULT "NORMAL"'),
                ('entry_hour', 'INTEGER DEFAULT 0'),
                ('trailing_stop_used', 'INTEGER DEFAULT 0')
            ]
            
            for col_name, col_def in new_columns:
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f'ALTER TABLE trades ADD COLUMN {col_name} {col_def}')
                        logger.info(f"Added column {col_name} to trades table")
                    except sqlite3.OperationalError as e:
                        logger.debug(f"Column {col_name} might already exist: {e}")
            
            # Add indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON trades(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_win_rate_analytics ON win_rate_analytics(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            logger.info(f"Enhanced win rate database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced database: {e}")
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get existing columns in a table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            conn.close()
            return columns
        except Exception:
            return []
    
    def get_recent_trades(self, symbol: str, hours: int = 24) -> List[EnhancedTrade]:
        """Get recent trades with enhanced data"""
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
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    
                    # Parse market conditions
                    market_conditions = {}
                    if row.get('market_conditions'):
                        try:
                            market_conditions = json.loads(row['market_conditions'])
                        except json.JSONDecodeError:
                            market_conditions = {}
                    
                    trades.append(EnhancedTrade(
                        timestamp=timestamp,
                        symbol=row['symbol'],
                        side=row['side'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        quantity=row['quantity'],
                        pnl_pct=row['pnl_pct'],
                        pnl_usd=row.get('pnl_usd', 0.0),
                        martingale_step=row.get('martingale_step', 0),
                        reason=row.get('reason', 'unknown'),
                        spread_at_entry=row.get('spread_at_entry', 0.0),
                        market_conditions=market_conditions,
                        confidence_score=row.get('confidence_score', 0.0),
                        market_regime=row.get('market_regime', 'NORMAL'),
                        entry_hour=row.get('entry_hour', timestamp.hour),
                        trailing_stop_used=bool(row.get('trailing_stop_used', 0))
                    ))
                except Exception as e:
                    logger.error(f"Error parsing trade row: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def get_win_rate_analytics(self, symbol: str, hours: int = 168) -> WinRateMetrics:
        """Get comprehensive win rate analytics"""
        try:
            trades = self.get_recent_trades(symbol, hours)
            
            if not trades:
                return WinRateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {}, 0, 0, 0, 0.0)
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'pnl_pct': t.pnl_pct,
                'timestamp': t.timestamp,
                'confidence_score': t.confidence_score,
                'market_regime': t.market_regime,
                'entry_hour': t.entry_hour,
                'trailing_stop_used': t.trailing_stop_used,
                'reason': t.reason
            } for t in trades])
            
            # Basic win rate metrics
            wins = df[df['pnl_pct'] > 0]
            losses = df[df['pnl_pct'] <= 0]
            
            total_trades = len(df)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            loss_rate = len(losses) / total_trades if total_trades > 0 else 0
            
            avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            
            win_loss_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else float('inf')
            
            total_profits = wins['pnl_pct'].sum()
            total_losses = abs(losses['pnl_pct'].sum())
            profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
            
            # Enhanced analytics
            win_rate_by_hour = self._calculate_win_rate_by_hour(df)
            win_rate_by_confidence = self._calculate_win_rate_by_confidence(df)
            win_rate_by_regime = self._calculate_win_rate_by_regime(df)
            
            consecutive_wins, consecutive_losses, max_consecutive_losses = self._calculate_streaks(df)
            
            trailing_stop_effectiveness = self._calculate_trailing_stop_effectiveness(df)
            
            return WinRateMetrics(
                total_trades=total_trades,
                win_rate=win_rate,
                loss_rate=loss_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                win_loss_ratio=win_loss_ratio,
                profit_factor=profit_factor,
                win_rate_by_hour=win_rate_by_hour,
                win_rate_by_confidence=win_rate_by_confidence,
                win_rate_by_regime=win_rate_by_regime,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                max_consecutive_losses=max_consecutive_losses,
                trailing_stop_effectiveness=trailing_stop_effectiveness
            )
            
        except Exception as e:
            logger.error(f"Failed to get win rate analytics: {e}")
            return WinRateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {}, 0, 0, 0, 0.0)
    
    def _calculate_win_rate_by_hour(self, df: pd.DataFrame) -> Dict[int, float]:
        """Calculate win rate by hour of day"""
        win_rates = {}
        for hour in range(24):
            hour_trades = df[df['entry_hour'] == hour]
            if len(hour_trades) > 0:
                wins = len(hour_trades[hour_trades['pnl_pct'] > 0])
                win_rates[hour] = wins / len(hour_trades)
        
        return win_rates
    
    def _calculate_win_rate_by_confidence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate win rate by confidence score ranges"""
        ranges = {
            'Low (0.0-0.6)': (0.0, 0.6),
            'Medium (0.6-0.8)': (0.6, 0.8),
            'High (0.8-1.0)': (0.8, 1.0)
        }
        
        win_rates = {}
        for label, (min_conf, max_conf) in ranges.items():
            conf_trades = df[(df['confidence_score'] >= min_conf) & (df['confidence_score'] < max_conf)]
            if len(conf_trades) > 0:
                wins = len(conf_trades[conf_trades['pnl_pct'] > 0])
                win_rates[label] = wins / len(conf_trades)
        
        return win_rates
    
    def _calculate_win_rate_by_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate win rate by market regime"""
        win_rates = {}
        for regime in ['LOW_VOL', 'NORMAL', 'HIGH_VOL']:
            regime_trades = df[df['market_regime'] == regime]
            if len(regime_trades) > 0:
                wins = len(regime_trades[regime_trades['pnl_pct'] > 0])
                win_rates[regime] = wins / len(regime_trades)
        
        return win_rates
    
    def _calculate_streaks(self, df: pd.DataFrame) -> Tuple[int, int, int]:
        """Calculate win/loss streaks"""
        if len(df) == 0:
            return 0, 0, 0
        
        df_sorted = df.sort_values('timestamp')
        
        current_wins = 0
        current_losses = 0
        max_losses = 0
        
        for _, row in df_sorted.iterrows():
            if row['pnl_pct'] > 0:
                current_wins += 1
                current_losses = 0
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return current_wins, current_losses, max_losses
    
    def _calculate_trailing_stop_effectiveness(self, df: pd.DataFrame) -> float:
        """Calculate effectiveness of trailing stops"""
        trailing_trades = df[df['trailing_stop_used'] == True]
        if len(trailing_trades) == 0:
            return 0.0
        
        trailing_profits = len(trailing_trades[
            (trailing_trades['reason'] == 'trailing_stop') & 
            (trailing_trades['pnl_pct'] > 0)
        ])
        return trailing_profits / len(trailing_trades) if len(trailing_trades) > 0 else 0.0

class WinRateOptimizedAnalyzer:
    """AI analyzer focused on win rate optimization"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def analyze_win_rate_optimization(self, metrics: WinRateMetrics, current_config: Dict) -> List[WinRateConfigUpdate]:
        """Analyze and optimize for win rate improvement"""
        
        if metrics.total_trades < 10:
            logger.info(f"Insufficient trades for win rate analysis: {metrics.total_trades} < 10")
            return []
        
        # Create win rate focused prompt
        prompt = self._create_win_rate_prompt(metrics, current_config)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_win_rate_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            ai_response = response.choices[0].message.content
            logger.info("Win rate optimization analysis completed")
            
            recommendations = self._parse_win_rate_recommendations(ai_response, metrics)
            validated_recommendations = self._validate_win_rate_recommendations(recommendations, current_config)
            
            return validated_recommendations
            
        except Exception as e:
            logger.error(f"Win rate analysis failed: {e}")
            return []
    
    def _get_win_rate_system_prompt(self) -> str:
        """System prompt focused on win rate optimization"""
        return """You are an expert quantitative trading analyst specializing in WIN RATE OPTIMIZATION for cryptocurrency scalping strategies.

Your PRIMARY OBJECTIVE: Maximize win rate while maintaining or improving risk-adjusted returns.

CORE PRINCIPLES:
1. WIN RATE IS PRIORITY: Target 65%+ win rate for scalping strategies
2. QUALITY OVER QUANTITY: Fewer, higher-probability trades beat high-frequency low-quality trades
3. RISK MANAGEMENT: Tight stops and selective entries improve win rate
4. MARKET ADAPTATION: Different regimes require different parameters
5. CONFIDENCE-BASED SIZING: Trade larger when confidence is higher

FOCUS AREAS FOR WIN RATE IMPROVEMENT:
- Entry selectivity (spread requirements, book depth, timing)
- Risk/reward optimization (tighter stops, appropriate targets)
- Market regime adaptation (volatility-based adjustments)
- Time-of-day filtering (avoid low-liquidity periods)
- Position sizing based on confidence scores

AVOID: Martingale systems, wide stops, revenge trading, overtrading during losses

Respond ONLY in valid JSON format with decimal numbers (no % signs)."""
    
    def _create_win_rate_prompt(self, metrics: WinRateMetrics, config: Dict) -> str:
        """Create win rate optimization focused prompt"""
        
        # Identify best and worst performing hours
        best_hour = max(metrics.win_rate_by_hour.items(), key=lambda x: x[1]) if metrics.win_rate_by_hour else (12, 0.5)
        worst_hour = min(metrics.win_rate_by_hour.items(), key=lambda x: x[1]) if metrics.win_rate_by_hour else (3, 0.3)
        
        return f"""
WIN RATE OPTIMIZATION ANALYSIS - CRYPTOCURRENCY SCALPING

CURRENT PERFORMANCE METRICS:
Win Rate: {metrics.win_rate:.1%} (Target: >65% for scalping)
Total Trades: {metrics.total_trades}
Risk/Reward Ratio: {metrics.win_loss_ratio:.2f}
Profit Factor: {metrics.profit_factor:.2f}
Max Consecutive Losses: {metrics.max_consecutive_losses} (Target: <4)

WIN RATE BREAKDOWN:
- Winning Trades: {int(metrics.win_rate * metrics.total_trades)}
- Losing Trades: {int(metrics.loss_rate * metrics.total_trades)}
- Average Win: {metrics.avg_win_pct:.3%}
- Average Loss: {metrics.avg_loss_pct:.3%}

PERFORMANCE BY TIME:
Best Hour: {best_hour[0]:02d}:00 UTC ({best_hour[1]:.1%} win rate)
Worst Hour: {worst_hour[0]:02d}:00 UTC ({worst_hour[1]:.1%} win rate)
Time Filter Enabled: {config.get('ENABLE_TIME_FILTER', False)}

PERFORMANCE BY CONFIDENCE:
{json.dumps(metrics.win_rate_by_confidence, indent=2) if metrics.win_rate_by_confidence else 'No confidence data available'}

PERFORMANCE BY MARKET REGIME:
{json.dumps(metrics.win_rate_by_regime, indent=2) if metrics.win_rate_by_regime else 'No regime data available'}

TRAILING STOP EFFECTIVENESS: {metrics.trailing_stop_effectiveness:.1%}

CURRENT CONFIGURATION:
Entry Filters:
- Min Spread: {config.get('MIN_SPREAD', 'N/A')}
- Min Book Depth: {config.get('MIN_BOOK_DEPTH', 'N/A')}
- Max Spread Volatility: {config.get('MAX_SPREAD_VOLATILITY', 'N/A')}

Risk Management:
- Profit Target: {config.get('PROFIT_TARGET', 'N/A')}
- Stop Loss: {config.get('STOP_LOSS', 'N/A')}
- Trailing Stops: {config.get('ENABLE_TRAILING_STOP', 'N/A')}

Timing Controls:
- Time Filter: {config.get('ENABLE_TIME_FILTER', 'N/A')}
- Cooldown Seconds: {config.get('COOLDOWN_SECONDS', 'N/A')}

Market Adaptation:
- Market Regime Detection: {config.get('ENABLE_MARKET_REGIME', 'N/A')}
- High Vol Threshold: {config.get('HIGH_VOL_THRESHOLD', 'N/A')}
- Low Vol Threshold: {config.get('LOW_VOL_THRESHOLD', 'N/A')}

Martingale System:
- Enabled: {config.get('MARTINGALE_ENABLED', 'N/A')}
- Multiplier: {config.get('MARTINGALE_MULTIPLIER', 'N/A')}
- Max Steps: {config.get('MARTINGALE_MAX_STEPS', 'N/A')}

WIN RATE OPTIMIZATION PRIORITIES:
1. If win rate < 60%: Focus on entry selectivity and stop-loss optimization
2. If consecutive losses > 3: Improve market regime detection
3. If profit factor < 1.5: Optimize risk/reward ratio
4. If time-based performance varies >20%: Enhance time filtering
5. If confidence correlation weak: Improve confidence scoring

RECOMMENDED PARAMETER ADJUSTMENTS:
Focus on parameters that directly impact win rate:
- MIN_SPREAD: Higher values = more selective entries = higher win rate
- STOP_LOSS: Tighter stops can improve win rate but need proper R/R balance
- PROFIT_TARGET: Should maintain good risk/reward while achievable
- MIN_BOOK_DEPTH: Higher depth = better execution = improved win rate
- COOLDOWN_SECONDS: Longer cooldowns = more selective timing
- MARTINGALE_ENABLED: Disable for pure win rate focus

JSON Response Format:
{{
    "win_rate_assessment": "Current {metrics.win_rate:.1%} win rate analysis",
    "primary_issues": ["list", "of", "main", "win", "rate", "problems"],
    "optimization_strategy": "Overall approach to improve win rate",
    "recommendations": [
        {{
            "parameter": "PARAMETER_NAME",
            "current_value": 0.005,
            "suggested_value": 0.007,
            "reason": "Specific win rate improvement justification",
            "confidence": 0.85,
            "expected_win_rate_impact": "Increase from 55% to 62%",
            "risk_level": "Low",
            "priority": "HIGH"
        }}
    ]
}}
"""
    
    def _parse_win_rate_recommendations(self, ai_response: str, metrics: WinRateMetrics) -> List[WinRateConfigUpdate]:
        """Parse AI response for win rate optimization"""
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in win rate analysis response")
                return []
            
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in win rate analysis: {e}")
                return []
            
            recommendations = []
            
            for i, rec in enumerate(data.get('recommendations', [])):
                try:
                    # Parse numeric values
                    old_value = self._parse_numeric_value(rec['current_value'])
                    new_value = self._parse_numeric_value(rec['suggested_value'])
                    confidence = self._parse_numeric_value(rec['confidence'])
                    
                    if old_value is None or new_value is None or confidence is None:
                        logger.warning(f"Skipping win rate recommendation {i+1} due to unparseable values")
                        continue
                    
                    # Validate parameter
                    parameter = rec.get('parameter', '').upper()
                    valid_win_rate_parameters = [
                        'MIN_SPREAD', 'STOP_LOSS', 'PROFIT_TARGET', 'MIN_BOOK_DEPTH',
                        'MAX_SPREAD_VOLATILITY', 'COOLDOWN_SECONDS', 'HIGH_VOL_THRESHOLD',
                        'LOW_VOL_THRESHOLD', 'MARTINGALE_ENABLED', 'MARTINGALE_MAX_STEPS'
                    ]
                    
                    if parameter not in valid_win_rate_parameters:
                        logger.warning(f"Invalid win rate parameter: {parameter}")
                        continue
                    
                    recommendations.append(WinRateConfigUpdate(
                        parameter=parameter,
                        old_value=old_value,
                        new_value=new_value,
                        reason=rec.get('reason', 'Win rate optimization'),
                        confidence=confidence,
                        expected_win_rate_impact=rec.get('expected_win_rate_impact', 'Not specified'),
                        risk_level=rec.get('risk_level', 'Medium'),
                        priority=rec.get('priority', 'MEDIUM')
                    ))
                    
                    logger.info(f"✅ Parsed win rate recommendation: {parameter} {old_value} -> {new_value}")
                    
                except Exception as e:
                    logger.error(f"Error parsing win rate recommendation {i+1}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse win rate recommendations: {e}")
            return []
    
    def _parse_numeric_value(self, value: Any) -> Optional[float]:
        """Parse numeric values from AI response"""
        if value is None:
            return None
            
        str_value = str(value).strip()
        if not str_value:
            return None
        
        try:
            # Handle percentages
            if '%' in str_value:
                return float(str_value.replace('%', '')) / 100
            
            # Handle other formatting
            str_value = str_value.replace('$', '').replace(',', '')
            return float(str_value)
            
        except (ValueError, TypeError):
            return None
    
    def _validate_win_rate_recommendations(self, recommendations: List[WinRateConfigUpdate], current_config: Dict) -> List[WinRateConfigUpdate]:
        """Validate recommendations with win rate focus"""
        validated = []
        
        for rec in recommendations:
            # Win rate focused validation
            if rec.parameter == 'MIN_SPREAD':
                if rec.new_value > 0.015:  # Max 1.5%
                    logger.warning(f"MIN_SPREAD too high for liquidity: {rec.new_value}")
                    continue
                elif rec.new_value < 0.002:  # Min 0.2%
                    logger.warning(f"MIN_SPREAD too low for selectivity: {rec.new_value}")
                    rec.new_value = 0.002
                    rec.confidence *= 0.8
            
            elif rec.parameter == 'STOP_LOSS':
                if rec.new_value > 0.01:  # Max 1% for scalping
                    logger.warning(f"STOP_LOSS too wide for scalping: {rec.new_value}")
                    rec.new_value = 0.01
                    rec.confidence *= 0.7
                elif rec.new_value < 0.002:  # Min 0.2%
                    logger.warning(f"STOP_LOSS too tight: {rec.new_value}")
                    rec.new_value = 0.002
                    rec.confidence *= 0.8
            
            elif rec.parameter == 'PROFIT_TARGET':
                # Ensure good risk/reward for win rate
                current_sl = current_config.get('STOP_LOSS', 0.005)
                risk_reward = rec.new_value / current_sl
                if risk_reward < 1.2:  # Minimum 1.2:1 R/R
                    logger.warning(f"Risk/reward too low: {risk_reward:.2f}")
                    rec.new_value = current_sl * 1.5  # 1.5:1 R/R
                    rec.confidence *= 0.8
            
            elif rec.parameter == 'MIN_BOOK_DEPTH':
                if rec.new_value > 10000:  # Max $10k depth requirement
                    logger.warning(f"MIN_BOOK_DEPTH too high: {rec.new_value}")
                    rec.new_value = 10000
                    rec.confidence *= 0.8
                elif rec.new_value < 500:  # Min $500
                    logger.warning(f"MIN_BOOK_DEPTH too low: {rec.new_value}")
                    rec.new_value = 500
                    rec.confidence *= 0.9
            
            elif rec.parameter == 'COOLDOWN_SECONDS':
                if rec.new_value > 1800:  # Max 30 minutes
                    logger.warning(f"COOLDOWN too long: {rec.new_value}s")
                    rec.new_value = 1800
                    rec.confidence *= 0.8
                elif rec.new_value < 60:  # Min 1 minute
                    logger.warning(f"COOLDOWN too short: {rec.new_value}s")
                    rec.new_value = 60
                    rec.confidence *= 0.7
            
            elif rec.parameter == 'MARTINGALE_ENABLED':
                # For win rate focus, prefer disabling martingale
                if rec.new_value > 0.5:  # Enabling martingale
                    logger.info(f"Martingale enabling detected - ensuring conservative settings")
                    rec.risk_level = "Medium"
            
            elif rec.parameter == 'MARTINGALE_MAX_STEPS':
                if rec.new_value > 3:  # Max 3 steps for win rate focus
                    logger.warning(f"MARTINGALE_MAX_STEPS too high for win rate: {rec.new_value}")
                    rec.new_value = 3
                    rec.confidence *= 0.7
                    rec.risk_level = "High"
            
            # Priority-based confidence adjustment
            if rec.priority == 'HIGH':
                min_confidence = 0.7
            elif rec.priority == 'MEDIUM':
                min_confidence = 0.6
            else:  # LOW
                min_confidence = 0.8  # Higher bar for low priority changes
            
            if rec.confidence >= min_confidence:
                validated.append(rec)
                logger.info(f"✅ Validated win rate recommendation: {rec.parameter} (confidence: {rec.confidence:.2f}, priority: {rec.priority})")
            else:
                logger.info(f"❌ Win rate recommendation below threshold: {rec.parameter} ({rec.confidence:.2f} < {min_confidence})")
        
        return validated

class EnhancedConfigManager:
    """Enhanced configuration manager with win rate focus"""
    
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
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle boolean values
                        if value.lower() in ['true', 'false']:
                            config[key] = value.lower() == 'true'
                        else:
                            try:
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
    
    def update_win_rate_config(self, symbol: str, updates: List[WinRateConfigUpdate]) -> bool:
        """Apply win rate optimization updates"""
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
            backup_file = self.backup_dir / f"{symbol.lower()}.env.winrate_backup.{timestamp}"
            with open(backup_file, 'w') as f:
                f.writelines(lines)
            logger.info(f"📄 Win rate config backup created: {backup_file}")
            
            # Sort updates by priority
            sorted_updates = sorted(updates, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x.priority])
            
            # Apply updates
            updated_count = 0
            for update in sorted_updates:
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{update.parameter}="):
                        # Format value based on parameter type
                        if update.parameter in ['COOLDOWN_SECONDS', 'MIN_BOOK_DEPTH']:
                            formatted_value = str(int(update.new_value))
                        elif update.parameter == 'MARTINGALE_ENABLED':
                            formatted_value = 'true' if update.new_value > 0.5 else 'false'
                        else:
                            formatted_value = str(update.new_value)
                        
                        lines[i] = f"{update.parameter}={formatted_value}\n"
                        
                        logger.info(f"🎯 WIN RATE OPTIMIZATION: {update.parameter}")
                        logger.info(f"   📊 Change: {update.old_value} → {update.new_value}")
                        logger.info(f"   🎯 Expected Impact: {update.expected_win_rate_impact}")
                        logger.info(f"   ⚠️ Risk: {update.risk_level} | Priority: {update.priority}")
                        logger.info(f"   💡 Reason: {update.reason}")
                        
                        updated_count += 1
                        break
                else:
                    # Parameter not found, add it
                    if update.parameter in ['COOLDOWN_SECONDS', 'MIN_BOOK_DEPTH']:
                        formatted_value = str(int(update.new_value))
                    elif update.parameter == 'MARTINGALE_ENABLED':
                        formatted_value = 'true' if update.new_value > 0.5 else 'false'
                    else:
                        formatted_value = str(update.new_value)
                    
                    lines.append(f"{update.parameter}={formatted_value}\n")
                    logger.info(f"➕ Added new win rate parameter: {update.parameter}={formatted_value}")
                    updated_count += 1
            
            # Write updated config with win rate header
            header = f"# WIN RATE OPTIMIZED by Enhanced AI Agent on {datetime.now().isoformat()}\n"
            header += f"# Applied {updated_count} win rate optimization updates\n"
            header += f"# Focus: Maximum win rate with controlled risk\n\n"
            
            with open(config_file, 'w') as f:
                f.write(header)
                f.writelines(lines)
            
            logger.info(f"🎉 Applied {updated_count} win rate optimizations to {symbol}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to update win rate config: {e}")
            return False

class EnhancedWinRateAgent:
    """Enhanced trading agent focused on win rate optimization"""
    
    def __init__(self, deepseek_api_key: str):
        self.db = EnhancedTradingDatabase()
        self.analyzer = WinRateOptimizedAnalyzer(deepseek_api_key)
        self.config_manager = EnhancedConfigManager()
    
    def analyze_and_optimize_win_rate(self, symbol: str, min_trades: int = 10) -> bool:
        """Analyze and optimize for maximum win rate"""
        
        logger.info(f"🎯 Starting WIN RATE optimization analysis for {symbol}")
        
        # Get win rate analytics
        metrics = self.db.get_win_rate_analytics(symbol, hours=168)  # 1 week
        
        if metrics.total_trades < min_trades:
            logger.info(f"⏳ Insufficient trades for win rate analysis: {metrics.total_trades} < {min_trades}")
            return False
        
        # Load current config
        current_config = self.config_manager.load_config(symbol)
        if not current_config:
            logger.error(f"❌ Could not load configuration for {symbol}")
            return False
        
        # Log current win rate performance
        logger.info(f"📊 Current Win Rate Analysis:")
        logger.info(f"   🎯 Win Rate: {metrics.win_rate:.1%} (Target: >65%)")
        logger.info(f"   📈 Risk/Reward: 1:{metrics.win_loss_ratio:.2f}")
        logger.info(f"   💰 Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"   🔻 Max Consecutive Losses: {metrics.max_consecutive_losses}")
        
        if metrics.win_rate_by_hour:
            best_hours = sorted(metrics.win_rate_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
            worst_hours = sorted(metrics.win_rate_by_hour.items(), key=lambda x: x[1])[:3]
            logger.info(f"   ⏰ Best Hours: {[(f'{h:02d}:00', f'{wr:.1%}') for h, wr in best_hours]}")
            logger.info(f"   ⏰ Worst Hours: {[(f'{h:02d}:00', f'{wr:.1%}') for h, wr in worst_hours]}")
        
        # Get AI recommendations for win rate optimization
        recommendations = self.analyzer.analyze_win_rate_optimization(metrics, current_config)
        
        if not recommendations:
            logger.info("💡 No win rate optimization recommendations from AI analysis")
            return False
        
        # Sort by priority and log recommendations
        high_priority = [r for r in recommendations if r.priority == 'HIGH']
        medium_priority = [r for r in recommendations if r.priority == 'MEDIUM']
        low_priority = [r for r in recommendations if r.priority == 'LOW']
        
        logger.info(f"🎯 Received {len(recommendations)} win rate optimization recommendations:")
        if high_priority:
            logger.info(f"   🔴 HIGH Priority ({len(high_priority)}): {[r.parameter for r in high_priority]}")
        if medium_priority:
            logger.info(f"   🟡 MEDIUM Priority ({len(medium_priority)}): {[r.parameter for r in medium_priority]}")
        if low_priority:
            logger.info(f"   🟢 LOW Priority ({len(low_priority)}): {[r.parameter for r in low_priority]}")
        
        # Apply updates
        success = self.config_manager.update_win_rate_config(symbol, recommendations)
        
        if success:
            logger.info(f"✅ Applied win rate optimization for {symbol}")
            
            # Log detailed changes
            for rec in recommendations:
                logger.info(f"🎯 {rec.parameter}: {rec.old_value} → {rec.new_value}")
                logger.info(f"   📊 Expected Impact: {rec.expected_win_rate_impact}")
                logger.info(f"   ⚠️ Risk Level: {rec.risk_level}")
                logger.info(f"   🎯 Priority: {rec.priority}")
            
            # Log to database
            for rec in recommendations:
                self._log_win_rate_optimization(symbol, rec, metrics)
            
            # Restart service if enabled
            auto_restart = os.getenv('AUTO_RESTART_SERVICES', 'true').lower() == 'true'
            if auto_restart:
                logger.info(f"🔄 Restarting trading service for {symbol} with win rate optimizations")
                os.system(f"sudo systemctl restart binance-scalper@{symbol.lower()}")
        
        return success
    
    def _log_win_rate_optimization(self, symbol: str, update: WinRateConfigUpdate, metrics: WinRateMetrics):
        """Log win rate optimization to database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Log to win_rate_analytics table
            cursor.execute('''
                INSERT INTO win_rate_analytics 
                (timestamp, symbol, period_hours, total_trades, win_rate, avg_win_pct, 
                 avg_loss_pct, profit_factor, max_consecutive_losses, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                168,  # 1 week analysis
                metrics.total_trades,
                metrics.win_rate,
                metrics.avg_win_pct,
                metrics.avg_loss_pct,
                metrics.profit_factor,
                metrics.max_consecutive_losses,
                json.dumps({
                    'parameter': update.parameter,
                    'old_value': update.old_value,
                    'new_value': update.new_value,
                    'expected_impact': update.expected_win_rate_impact,
                    'priority': update.priority
                })
            ))
            
            # Also log to config_changes table with win rate flag
            cursor.execute('''
                INSERT INTO config_changes 
                (timestamp, symbol, parameter, old_value, new_value, reason, confidence, 
                 statistical_significance, model_used, expected_impact, priority, win_rate_focused)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                update.parameter,
                update.old_value,
                update.new_value,
                update.reason,
                update.confidence,
                1.0,  # High significance for win rate focus
                'deepseek-chat-winrate',
                update.expected_win_rate_impact,
                update.priority,
                1  # Win rate focused flag
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log win rate optimization: {e}")
    
    def get_win_rate_report(self, symbol: str, hours: int = 168) -> Dict:
        """Generate comprehensive win rate report"""
        try:
            metrics = self.db.get_win_rate_analytics(symbol, hours)
            
            if metrics.total_trades == 0:
                return {'error': f'No trades found for {symbol} in last {hours//24} days'}
            
            # Calculate win rate grade
            if metrics.win_rate >= 0.7:
                grade = "A"
                grade_color = "🟢"
            elif metrics.win_rate >= 0.6:
                grade = "B"
                grade_color = "🟡"
            elif metrics.win_rate >= 0.5:
                grade = "C"
                grade_color = "🟠"
            else:
                grade = "D"
                grade_color = "🔴"
            
            # Best and worst hours
            best_hour = max(metrics.win_rate_by_hour.items(), key=lambda x: x[1]) if metrics.win_rate_by_hour else (12, 0.5)
            worst_hour = min(metrics.win_rate_by_hour.items(), key=lambda x: x[1]) if metrics.win_rate_by_hour else (3, 0.3)
            
            report = {
                'symbol': symbol,
                'period_days': hours // 24,
                'overall_grade': f"{grade_color} Grade {grade}",
                'win_rate': f"{metrics.win_rate:.1%}",
                'total_trades': metrics.total_trades,
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'risk_reward_ratio': f"1:{metrics.win_loss_ratio:.2f}",
                'max_consecutive_losses': metrics.max_consecutive_losses,
                'best_trading_hour': f"{best_hour[0]:02d}:00 UTC ({best_hour[1]:.1%})",
                'worst_trading_hour': f"{worst_hour[0]:02d}:00 UTC ({worst_hour[1]:.1%})",
                'trailing_stop_effectiveness': f"{metrics.trailing_stop_effectiveness:.1%}",
                'win_rate_by_confidence': metrics.win_rate_by_confidence,
                'win_rate_by_regime': metrics.win_rate_by_regime,
                'recommendations': self._generate_quick_recommendations(metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate win rate report: {e}")
            return {'error': str(e)}
    
    def _generate_quick_recommendations(self, metrics: WinRateMetrics) -> List[str]:
        """Generate quick win rate improvement recommendations"""
        recommendations = []
        
        if metrics.win_rate < 0.6:
            recommendations.append("🎯 Increase entry selectivity - raise MIN_SPREAD requirement")
        
        if metrics.max_consecutive_losses > 3:
            recommendations.append("🛑 Improve stop-loss strategy - consider tighter stops")
        
        if metrics.profit_factor < 1.5:
            recommendations.append("⚖️ Optimize risk/reward ratio - adjust targets and stops")
        
        if metrics.trailing_stop_effectiveness < 0.3:
            recommendations.append("📈 Review trailing stop strategy - may need adjustment")
        
        # Check confidence correlation
        if metrics.win_rate_by_confidence:
            high_conf_wr = metrics.win_rate_by_confidence.get('High (0.8-1.0)', 0)
            low_conf_wr = metrics.win_rate_by_confidence.get('Low (0.0-0.6)', 0)
            if high_conf_wr - low_conf_wr < 0.2:  # Less than 20% difference
                recommendations.append("🧠 Improve confidence scoring - correlation is weak")
        
        # Check regime performance
        if metrics.win_rate_by_regime:
            high_vol_wr = metrics.win_rate_by_regime.get('HIGH_VOL', 0)
            normal_wr = metrics.win_rate_by_regime.get('NORMAL', 0)
            if high_vol_wr > 0 and normal_wr > high_vol_wr + 0.15:  # 15% worse in high vol
                recommendations.append("🌡️ Avoid high volatility periods - enable stricter regime filtering")
        
        if not recommendations:
            recommendations.append("✅ Win rate performance looks good - continue monitoring")
        
        return recommendations

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
    """Main function with win rate optimization focus"""
    
    # Environment validation
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        logger.error("❌ DEEPSEEK_API_KEY not found in environment")
        return
    
    # Configuration
    config_dir = os.getenv('CONFIG_DIR', '/home/ubuntu/binance_scalper/configs')
    analysis_interval = int(os.getenv('ANALYSIS_INTERVAL_HOURS', '6'))  # More frequent for win rate focus
    min_trades = int(os.getenv('MIN_TRADES_FOR_ANALYSIS', '10'))  # Lower threshold for win rate analysis
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
        # Fallback to environment variable or default symbols
        env_symbols = os.getenv('SYMBOLS', '')
        if env_symbols:
            symbols = [s.strip().upper() for s in env_symbols.split(',')]
            logger.info(f"📝 Using symbols from environment: {symbols}")
        else:
            # Use default symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'ARBUSDT', 'SHIBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT']
            logger.info(f"🎯 Using default symbols: {symbols}")
    
    if not symbols:
        logger.error("❌ No valid symbols found. Please check your config directory or set SYMBOLS environment variable.")
        return
    
    logger.info(f"""
🎯 Enhanced AI Trading Agent - WIN RATE OPTIMIZATION
   📊 Primary Objective: Maximize Win Rate (Target: >65%)
   📈 Active Symbols ({len(symbols)}): {symbols}
   ⏱️ Analysis Interval: {analysis_interval}h
   📊 Min Trades for Analysis: {min_trades}
   🔧 Auto-update: {auto_update}
   🤖 AI Model: DeepSeek Chat - Win Rate Focused
   
   🎯 Win Rate Optimization Features:
      • Confidence-based position sizing analysis
      • Market regime performance tracking
      • Time-of-day win rate analytics
      • Order book depth correlation
      • Trailing stop effectiveness measurement
      • Risk/reward optimization
      • Consecutive loss pattern detection
      
   📊 Enhanced Analytics:
      • Win rate by hour of day
      • Performance by confidence score
      • Market regime effectiveness
      • Parameter correlation analysis
""")
    
    agent = EnhancedWinRateAgent(deepseek_api_key)
    
    # Initial validation and reporting
    logger.info("🔍 Performing initial win rate assessment...")
    for symbol in symbols[:3]:  # Check first 3 symbols as examples
        try:
            report = agent.get_win_rate_report(symbol, hours=168)
            if 'error' not in report:
                logger.info(f"   {symbol}: {report['overall_grade']} - {report['win_rate']} win rate ({report['total_trades']} trades)")
            else:
                logger.info(f"   {symbol}: {report['error']}")
        except Exception as e:
            logger.warning(f"   {symbol}: Could not generate report - {e}")
    
    # Main analysis loop
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
                logger.info(f"🎯 Win Rate Analysis for {symbol}...")
                
                # Check if config file still exists before analysis
                config_file = Path(config_dir) / f"{symbol.lower()}.env"
                if not config_file.exists():
                    logger.warning(f"⚠️  Config file missing for {symbol}, skipping...")
                    continue
                
                if auto_update:
                    success = agent.analyze_and_optimize_win_rate(symbol, min_trades=min_trades)
                    if success:
                        logger.info(f"✅ Win rate optimization completed for {symbol}")
                        
                        # Show updated win rate report
                        try:
                            report = agent.get_win_rate_report(symbol, hours=24)
                            if 'error' not in report:
                                logger.info(f"📊 Updated performance: {report['overall_grade']} - {report['win_rate']} win rate")
                        except Exception as e:
                            logger.debug(f"Could not generate updated report: {e}")
                    else:
                        logger.info(f"⏸️  No win rate optimization needed for {symbol}")
                else:
                    # Analysis only mode with detailed reporting
                    try:
                        report = agent.get_win_rate_report(symbol, hours=168)
                        if 'error' not in report:
                            logger.info(f"📊 Analysis-only mode for {symbol}:")
                            logger.info(f"   {report['overall_grade']} - {report['win_rate']} win rate")
                            logger.info(f"   Risk/Reward: {report['risk_reward_ratio']}")
                            logger.info(f"   Profit Factor: {report['profit_factor']}")
                            logger.info(f"   Best Hour: {report['best_trading_hour']}")
                            if report['recommendations']:
                                logger.info(f"   Suggestions: {report['recommendations'][0]}")
                        else:
                            logger.info(f"📊 Analysis-only: {symbol} - {report['error']}")
                    except Exception as e:
                        logger.warning(f"Analysis failed for {symbol}: {e}")
                
                time.sleep(30)  # Pause between symbols
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Enhanced Win Rate Optimization Agent...")
            break
        except Exception as e:
            logger.error(f"❌ Error in win rate analysis cycle: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info(f"⏳ Waiting {analysis_interval} hours for next win rate optimization cycle...")
        time.sleep(analysis_interval * 3600)

if __name__ == "__main__":
    main()
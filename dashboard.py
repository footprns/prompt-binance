#!/usr/bin/env python3
"""
Trading Dashboard - Fixed Version with Correct Paths
"""

from flask import Flask, render_template, jsonify
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class DashboardData:
    def __init__(self, db_path="/home/ubuntu/binance_scalper/trading_data.db"):
        self.db_path = db_path
        # Use full paths to system commands
        self.systemctl_path = '/usr/bin/systemctl'
        self.pgrep_path = '/usr/bin/pgrep'
        
        # Verify paths exist
        if not os.path.exists(self.systemctl_path):
            self.systemctl_path = '/bin/systemctl'
        if not os.path.exists(self.pgrep_path):
            self.pgrep_path = '/bin/pgrep'
    
    def get_performance(self, symbol, hours=168):
        """Get performance data for symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                conn.close()
                return {'error': 'No trading data available yet. Start trading to see performance.'}
            
            df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            '''.format(hours), conn, params=(symbol,))
            
            conn.close()
            
            if df.empty:
                return {'error': f'No trades found for {symbol} in the last {hours//24} days'}
            
            # Calculate metrics
            df['cumulative_pnl'] = df['pnl_pct'].cumsum() * 100
            win_rate = len(df[df['pnl_pct'] > 0]) / len(df)
            total_pnl = df['pnl_pct'].sum() * 100
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L %',
                line=dict(color='blue' if total_pnl >= 0 else 'red', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f'{symbol} Performance ({len(df)} trades)',
                xaxis_title='Time',
                yaxis_title='Cumulative P&L %',
                height=400,
                template='plotly_white'
            )
            
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'chart': graphJSON,
                'total_trades': len(df),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': df[df['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if len(df[df['pnl_pct'] > 0]) > 0 else 0,
                'avg_loss': df[df['pnl_pct'] < 0]['pnl_pct'].mean() * 100 if len(df[df['pnl_pct'] < 0]) > 0 else 0,
                'last_trade': df.iloc[-1]['timestamp'] if len(df) > 0 else 'None'
            }
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            return {'error': f'Database error: {str(e)}'}
    
    def get_system_status(self):
        """Get system status with multiple fallback methods"""
        try:
            # Check AI agent status
            ai_status = self._check_service_status('binance-ai-agent')
            
            # Check active trading services
            active_traders = []
            trader_details = {}
            
            for symbol in ['btcusdt', 'ethusdt', 'arbusdt', 'shibusdt', 'adausdt', 'dogeusdt', 'xrpusdt', 'bnbusdt', 'pepeusdt']:
            # Added PEPEUSDT
                service_name = f'binance-scalper@{symbol}'
                status = self._check_service_status(service_name)
                
                if status == 'active':
                    active_traders.append(symbol.upper())
                    trader_details[symbol.upper()] = 'Running'
                elif status == 'failed':
                    trader_details[symbol.upper()] = 'Failed'
                elif status == 'inactive':
                    trader_details[symbol.upper()] = 'Stopped'
                else:
                    trader_details[symbol.upper()] = f'Unknown ({status})'
            
            # Get additional info
            last_analysis = self._get_last_analysis_time()
            trade_counts = self._get_recent_trade_counts()
            
            return {
                'ai_agent_status': ai_status,
                'last_analysis': last_analysis,
                'active_traders': active_traders,
                'trader_details': trader_details,
                'trade_counts': trade_counts,
                'dashboard_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'systemctl_path': self.systemctl_path
            }
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {
                'ai_agent_status': f'Error: {str(e)}',
                'last_analysis': 'Status check failed',
                'active_traders': [],
                'trader_details': {},
                'trade_counts': {},
                'dashboard_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'systemctl_path': self.systemctl_path
            }
    
    def _check_service_status(self, service_name):
        """Check service status with multiple methods"""
        # Method 1: Try systemctl with full path
        try:
            if os.path.exists(self.systemctl_path):
                result = subprocess.run(
                    [self.systemctl_path, 'is-active', service_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env={'PATH': '/usr/bin:/bin:/usr/local/bin'}
                )
                
                status = result.stdout.strip()
                if status in ['active', 'inactive', 'failed']:
                    return status
                    
        except Exception as e:
            logger.error(f"Method 1 failed for {service_name}: {e}")
        
        # Method 2: Try with different environment
        try:
            result = subprocess.run(
                ['sudo', '/usr/bin/systemctl', 'is-active', service_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            status = result.stdout.strip()
            if status in ['active', 'inactive', 'failed']:
                return status
                
        except Exception as e:
            logger.error(f"Method 2 failed for {service_name}: {e}")
        
        # Method 3: Check process directly
        try:
            if service_name == 'binance-ai-agent':
                search_term = 'ai_trading_agent.py'
            else:
                # Extract symbol from service name like binance-scalper@btcusdt
                symbol = service_name.split('@')[1] if '@' in service_name else 'scalper'
                search_term = f'scalper.*{symbol}'
            
            if os.path.exists(self.pgrep_path):
                result = subprocess.run(
                    [self.pgrep_path, '-f', search_term],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                return 'running' if result.returncode == 0 else 'stopped'
            
        except Exception as e:
            logger.error(f"Method 3 failed for {service_name}: {e}")
        
        # Method 4: Check if log files are being written
        try:
            import time
            log_path = f"/var/log/syslog"
            if os.path.exists(log_path):
                # Check if service logged anything in last 5 minutes
                five_min_ago = time.time() - 300
                stat = os.stat(log_path)
                if stat.st_mtime > five_min_ago:
                    # Check if our service appears in recent logs
                    with open(log_path, 'r') as f:
                        # Read last 100 lines
                        lines = f.readlines()[-100:]
                        for line in lines:
                            if service_name in line:
                                if 'started' in line.lower() or 'running' in line.lower():
                                    return 'likely_running'
                                elif 'stopped' in line.lower() or 'failed' in line.lower():
                                    return 'likely_stopped'
            
        except Exception as e:
            logger.error(f"Method 4 failed for {service_name}: {e}")
        
        return 'unknown'
    
    def _get_last_analysis_time(self):
        """Get last analysis time from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if config_changes table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='config_changes'")
            if not cursor.fetchone():
                conn.close()
                return 'No analysis data yet'
            
            cursor.execute('''
                SELECT timestamp FROM config_changes 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Parse and format timestamp
                try:
                    dt = datetime.fromisoformat(result[0])
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return result[0]
            else:
                return 'No analysis performed yet'
                
        except Exception as e:
            return f'Database error: {str(e)}'
    
    def _get_recent_trade_counts(self):
        """Get recent trade counts for each symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                conn.close()
                return {}
            
            cursor.execute('''
                SELECT symbol, COUNT(*) as count
                FROM trades 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY symbol
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return {row[0]: row[1] for row in results}
            
        except Exception as e:
            logger.error(f"Error getting trade counts: {e}")
            return {}

dashboard_data = DashboardData()

@app.route('/')
def dashboard():
    """Trading dashboard"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { 
                background: rgba(255,255,255,0.95); 
                color: #2c3e50; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .symbol-tabs { margin: 20px 0; }
            .tab { 
                display: inline-block; 
                padding: 12px 24px; 
                background: rgba(255,255,255,0.9); 
                color: #2c3e50; 
                margin-right: 8px; 
                cursor: pointer; 
                border-radius: 25px;
                transition: all 0.3s ease;
            }
            .tab.active { background: #3498db; color: white; }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }
            .metric { 
                padding: 20px; 
                background: rgba(255,255,255,0.95); 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .metric h3 { margin: 0 0 10px 0; color: #2c3e50; }
            .metric .value { font-size: 24px; font-weight: bold; }
            .positive { color: #27ae60; }
            .negative { color: #e74c3c; }
            .chart, .status { 
                background: rgba(255,255,255,0.95); 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .status-item { 
                display: flex; 
                justify-content: space-between; 
                padding: 8px 0; 
                border-bottom: 1px solid #ecf0f1;
            }
            .status-running { color: #27ae60; font-weight: bold; }
            .status-stopped { color: #e74c3c; font-weight: bold; }
            .status-unknown { color: #f39c12; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Trading Dashboard</h1>
                <p>Real-time cryptocurrency scalping performance</p>
            </div>
            
            <div class="symbol-tabs">
                <div class="tab active" onclick="loadSymbol('BTCUSDT')">BTC/USDT</div>
                <div class="tab" onclick="loadSymbol('ETHUSDT')">ETH/USDT</div>
                <div class="tab" onclick="loadSymbol('ARBUSDT')">ARB/USDT</div>
                <div class="tab" onclick="loadSymbol('SHIBUSDT')">SHIB/USDT</div>
                <div class="tab" onclick="loadSymbol('ADAUSDT')">ADA/USDT</div>
                <div class="tab" onclick="loadSymbol('DOGEUSDT')">DOGE/USDT</div>
                <div class="tab" onclick="loadSymbol('XRPUSDT')">XRP/USDT</div>
                <div class="tab" onclick="loadSymbol('BNBUSDT')">BNB/USDT</div>
                <div class="tab" onclick="loadSymbol('PEPEUSDT')">PEPE/USDT</div>
                <div class="tab" onclick="loadSymbol('TRONUSDT')">TRON/USDT</div>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Total Trades</h3>
                    <div class="value" id="total-trades">Loading...</div>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <div class="value" id="win-rate">Loading...</div>
                </div>
                <div class="metric">
                    <h3>Total P&L</h3>
                    <div class="value" id="total-pnl">Loading...</div>
                </div>
                <div class="metric">
                    <h3>Avg Win</h3>
                    <div class="value" id="avg-win">Loading...</div>
                </div>
                <div class="metric">
                    <h3>Avg Loss</h3>
                    <div class="value" id="avg-loss">Loading...</div>
                </div>
            </div>
            
            <div class="chart">
                <div id="chart"></div>
            </div>
            
            <div class="status">
                <h3>System Status</h3>
                <div id="status-info">Loading...</div>
            </div>
        </div>

        <script>
            let currentSymbol = 'BTCUSDT';
            
            function loadSymbol(symbol) {
                currentSymbol = symbol;
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                event.target.classList.add('active');
                loadData(symbol);
            }
            
            function loadData(symbol = 'BTCUSDT') {
                fetch(`/api/performance/${symbol}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('total-trades').textContent = '0';
                            document.getElementById('win-rate').textContent = 'No Data';
                            document.getElementById('total-pnl').textContent = 'No Data';
                            document.getElementById('avg-win').textContent = 'No Data';
                            document.getElementById('avg-loss').textContent = 'No Data';
                            document.getElementById('chart').innerHTML = `<p style="text-align: center; color: #7f8c8d;">${data.error}</p>`;
                            return;
                        }
                        
                        document.getElementById('total-trades').textContent = data.total_trades;
                        document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(1) + '%';
                        
                        const pnlElement = document.getElementById('total-pnl');
                        pnlElement.textContent = data.total_pnl.toFixed(2) + '%';
                        pnlElement.className = 'value ' + (data.total_pnl >= 0 ? 'positive' : 'negative');
                        
                        document.getElementById('avg-win').textContent = data.avg_win.toFixed(2) + '%';
                        document.getElementById('avg-loss').textContent = data.avg_loss.toFixed(2) + '%';
                        
                        if (data.chart) {
                            Plotly.newPlot('chart', JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            function loadStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        let html = '';
                        
                        // AI Agent
                        const aiClass = data.ai_agent_status === 'active' ? 'status-running' : 
                                       data.ai_agent_status.includes('Error') ? 'status-unknown' : 'status-stopped';
                        html += `<div class="status-item">
                            <span><strong>AI Agent:</strong></span>
                            <span class="${aiClass}">${data.ai_agent_status}</span>
                        </div>`;
                        
                        // Last Analysis
                        html += `<div class="status-item">
                            <span><strong>Last Analysis:</strong></span>
                            <span>${data.last_analysis}</span>
                        </div>`;
                        
                        // Traders
                        if (data.trader_details) {
                            Object.entries(data.trader_details).forEach(([symbol, status]) => {
                                const statusClass = status === 'Running' ? 'status-running' : 
                                                   status.includes('Unknown') ? 'status-unknown' : 'status-stopped';
                                html += `<div class="status-item">
                                    <span><strong>${symbol}:</strong></span>
                                    <span class="${statusClass}">${status}</span>
                                </div>`;
                            });
                        }
                        
                        // Trade counts
                        if (data.trade_counts && Object.keys(data.trade_counts).length > 0) {
                            html += '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ecf0f1;"><strong>24h Trades:</strong></div>';
                            Object.entries(data.trade_counts).forEach(([symbol, count]) => {
                                html += `<div class="status-item">
                                    <span>${symbol}:</span>
                                    <span>${count}</span>
                                </div>`;
                            });
                        }
                        
                        html += `<div class="status-item">
                            <span><strong>Updated:</strong></span>
                            <span>${data.dashboard_time}</span>
                        </div>`;
                        
                        document.getElementById('status-info').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('status-info').innerHTML = '<span style="color: #e74c3c;">Status check failed</span>';
                    });
            }
            
            // Initial load
            loadData();
            loadStatus();
            
            // Auto refresh
            setInterval(() => {
                loadData(currentSymbol);
                loadStatus();
            }, 30000);
        </script>
    </body>
    </html>
    '''

@app.route('/api/performance/<symbol>')
def get_performance(symbol):
    return jsonify(dashboard_data.get_performance(symbol))

@app.route('/api/status')
def get_status():
    return jsonify(dashboard_data.get_system_status())

if __name__ == '__main__':
    logger.info("Starting dashboard on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)

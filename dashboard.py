#!/usr/bin/env python3
"""
Trading Dashboard - Complete Enhanced Version with Dynamic Service Discovery
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
import re

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
    
    def discover_active_scalper_services(self):
        """Dynamically discover all running scalper services"""
        active_services = {}
        
        try:
            # Method 1: Use systemctl to list all services
            result = subprocess.run(
                [self.systemctl_path, 'list-units', '--type=service', '--state=active', '--no-pager'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    # Look for binance-scalper@ services
                    match = re.search(r'binance-scalper@(\w+)\.service', line)
                    if match:
                        symbol = match.group(1).upper()
                        active_services[symbol] = 'Running'
            
            # Method 2: Check for specific common symbols if method 1 fails
            if not active_services:
                common_symbols = [
                    'btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'xrpusdt', 
                    'dogeusdt', 'shibusdt', 'arbusdt', 'pepeusdt', 'ltcusdt',
                    'solusdt', 'maticusdt', 'avaxusdt', 'linkusdt', 'dotusdt'
                ]
                
                for symbol in common_symbols:
                    service_name = f'binance-scalper@{symbol}'
                    status = self._check_service_status(service_name)
                    if status in ['active', 'running']:
                        active_services[symbol.upper()] = 'Running'
                    elif status in ['failed', 'inactive']:
                        active_services[symbol.upper()] = status.title()
            
            # Method 3: Check configs directory for additional symbols
            try:
                configs_dir = "/home/ubuntu/binance_scalper/configs"
                if os.path.exists(configs_dir):
                    for config_file in os.listdir(configs_dir):
                        if config_file.endswith('.env') and config_file != '.env_template':
                            symbol = config_file.replace('.env', '').upper()
                            if symbol not in active_services:
                                service_name = f'binance-scalper@{symbol.lower()}'
                                status = self._check_service_status(service_name)
                                if status in ['active', 'running']:
                                    active_services[symbol] = 'Running'
                                elif status in ['failed', 'inactive', 'stopped']:
                                    active_services[symbol] = status.title()
            except Exception as e:
                logger.debug(f"Could not check configs directory: {e}")
            
        except Exception as e:
            logger.error(f"Error discovering services: {e}")
        
        return active_services
    
    def get_system_status(self):
        """Get system status with dynamic service discovery"""
        try:
            # Check AI agent status
            ai_status = self._check_service_status('binance-ai-agent')
            
            # Discover all active trading services dynamically
            trader_details = self.discover_active_scalper_services()
            active_traders = [symbol for symbol, status in trader_details.items() if status == 'Running']
            
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
                'systemctl_path': self.systemctl_path,
                'total_services': len(trader_details)
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
                'systemctl_path': self.systemctl_path,
                'total_services': 0
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
            logger.debug(f"Method 1 failed for {service_name}: {e}")
        
        # Method 2: Try with sudo
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
            logger.debug(f"Method 2 failed for {service_name}: {e}")
        
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
                return 'active' if result.returncode == 0 else 'inactive'
            
        except Exception as e:
            logger.debug(f"Method 3 failed for {service_name}: {e}")
        
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
    """Trading dashboard with dynamic service discovery"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Dashboard - Enhanced</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { 
                background: rgba(255,255,255,0.95); 
                color: #2c3e50; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .symbol-tabs { margin: 20px 0; display: flex; flex-wrap: wrap; gap: 8px; }
            .tab { 
                display: inline-block; 
                padding: 8px 16px; 
                background: rgba(255,255,255,0.9); 
                color: #2c3e50; 
                cursor: pointer; 
                border-radius: 20px;
                transition: all 0.3s ease;
                font-size: 12px;
                min-width: 80px;
                text-align: center;
            }
            .tab.active { background: #3498db; color: white; }
            .tab:hover { background: #2980b9; color: white; }
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
            .status-failed { color: #e74c3c; font-weight: bold; }
            .status-inactive { color: #f39c12; font-weight: bold; }
            .status-unknown { color: #95a5a6; font-weight: bold; }
            .service-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
                gap: 8px; 
                margin: 10px 0; 
            }
            .service-card { 
                background: #f8f9fa; 
                padding: 8px; 
                border-radius: 8px; 
                text-align: center; 
                border-left: 4px solid #ddd;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .service-card:hover { background: #e9ecef; }
            .service-card.running { border-left-color: #27ae60; }
            .service-card.failed { border-left-color: #e74c3c; }
            .service-card.inactive { border-left-color: #f39c12; }
            .service-summary {
                background: #e8f4fd;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                border-left: 4px solid #3498db;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Trading Dashboard - Enhanced</h1>
                <p>Real-time cryptocurrency scalping performance with dynamic service discovery</p>
                <div id="service-summary" class="service-summary"></div>
            </div>
            
            <div class="symbol-tabs" id="symbol-tabs">
                <!-- Tabs will be populated dynamically -->
                <div class="tab active" onclick="loadSymbol('BTCUSDT')">BTC</div>
                <div class="tab" onclick="loadSymbol('ETHUSDT')">ETH</div>
                <div class="tab" onclick="loadSymbol('BNBUSDT')">BNB</div>
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
            let availableSymbols = [];
            
            function createSymbolTabs(symbols) {
                const tabsContainer = document.getElementById('symbol-tabs');
                tabsContainer.innerHTML = '';
                
                symbols.forEach((symbol, index) => {
                    const tab = document.createElement('div');
                    tab.className = `tab ${index === 0 ? 'active' : ''}`;
                    tab.textContent = symbol.replace('USDT', '');
                    tab.onclick = () => loadSymbol(symbol);
                    tabsContainer.appendChild(tab);
                });
                
                if (symbols.length > 0) {
                    currentSymbol = symbols[0];
                }
            }
            
            function loadSymbol(symbol) {
                currentSymbol = symbol;
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                event.target.classList.add('active');
                loadData(symbol);
            }
            
            function loadData(symbol) {
                if (!symbol) return;
                
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
                    .catch(error => {
                        console.error('Error loading data:', error);
                        document.getElementById('chart').innerHTML = '<p style="text-align: center; color: #e74c3c;">Error loading chart data</p>';
                    });
            }
            
            function loadStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        // Update service summary
                        const summaryElement = document.getElementById('service-summary');
                        const runningCount = data.active_traders ? data.active_traders.length : 0;
                        const totalCount = data.total_services || 0;
                        summaryElement.innerHTML = `
                            <strong>📊 Services:</strong> ${runningCount}/${totalCount} running | 
                            <strong>🤖 AI Agent:</strong> ${data.ai_agent_status} | 
                            <strong>🕒 Last Analysis:</strong> ${data.last_analysis}
                        `;
                        
                        // Update symbol tabs based on discovered services
                        const allSymbols = Object.keys(data.trader_details || {});
                        if (allSymbols.length > 0 && JSON.stringify(allSymbols.sort()) !== JSON.stringify(availableSymbols)) {
                            availableSymbols = allSymbols.sort();
                            createSymbolTabs(availableSymbols);
                            if (availableSymbols.includes(currentSymbol)) {
                                loadData(currentSymbol);
                            } else if (availableSymbols.length > 0) {
                                currentSymbol = availableSymbols[0];
                                loadData(currentSymbol);
                            }
                        }
                        
                        let html = '';
                        
                        // AI Agent status
                        const aiClass = data.ai_agent_status === 'active' ? 'status-running' : 
                                       data.ai_agent_status && data.ai_agent_status.includes('Error') ? 'status-unknown' : 'status-stopped';
                        html += `<div class="status-item">
                            <span><strong>🤖 AI Agent:</strong></span>
                            <span class="${aiClass}">${data.ai_agent_status || 'Unknown'}</span>
                        </div>`;
                        
                        // Service grid
                        if (data.trader_details && Object.keys(data.trader_details).length > 0) {
                            html += '<div style="margin-top: 15px;"><strong>📈 Trading Services:</strong></div>';
                            html += '<div class="service-grid">';
                            
                            Object.entries(data.trader_details).forEach(([symbol, status]) => {
                                const statusClass = status === 'Running' ? 'running' : 
                                                   status === 'Failed' ? 'failed' : 'inactive';
                                const tradeCount = data.trade_counts ? (data.trade_counts[symbol] || 0) : 0;
                                html += `<div class="service-card ${statusClass}" onclick="loadSymbol('${symbol}')">
                                    <div style="font-weight: bold; font-size: 12px;">${symbol}</div>
                                    <div style="font-size: 10px; color: #666;">${status}</div>
                                    <div style="font-size: 10px; color: #666;">${tradeCount} trades/24h</div>
                                </div>`;
                            });
                            
                            html += '</div>';
                        } else {
                            html += '<div style="margin-top: 15px; color: #666; text-align: center;">No trading services discovered</div>';
                        }
                        
                        html += `<div class="status-item" style="border-bottom: none; margin-top: 15px;">
                            <span><strong>🕒 Updated:</strong></span>
                            <span>${data.dashboard_time || 'Unknown'}</span>
                        </div>`;
                        
                        document.getElementById('status-info').innerHTML = html;
                    })
                    .catch(error => {
                        console.error('Error loading status:', error);
                        document.getElementById('status-info').innerHTML = '<span style="color: #e74c3c;">❌ Status check failed</span>';
                    });
            }
            
            // Initial load
            loadStatus();
            
            // Auto refresh every 30 seconds
            setInterval(() => {
                loadStatus();
                if (currentSymbol) {
                    loadData(currentSymbol);
                }
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
    logger.info("Starting enhanced dashboard on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
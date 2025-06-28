"""
Reporting module for generating P&L reports and trade statistics.
"""
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz
from pathlib import Path

from core.models import Portfolio, Position, Trade
from core.config_manager import ConfigManager

class PortfolioReporter:
    """Generates reports and statistics for the trading portfolio."""
    
    def __init__(self, config: ConfigManager, portfolio: Portfolio):
        """Initialize the portfolio reporter.
        
        Args:
            config: Configuration manager instance
            portfolio: Portfolio instance to generate reports for
        """
        self.config = config
        self.portfolio = portfolio
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = logging.getLogger(__name__)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate a daily trading report.
        
        Returns:
            dict: Dictionary containing the report data
        """
        try:
            # Get today's date in IST
            today = datetime.now(self.ist_tz).date()
            
            # Get all trades for today
            today_trades = [
                t for t in self.portfolio.get_all_trades() 
                if t.exit_time and t.exit_time.date() == today
            ]
            
            # Calculate statistics
            total_trades = len(today_trades)
            winning_trades = [t for t in today_trades if t.pnl > 0]
            losing_trades = [t for t in today_trades if t.pnl <= 0]
            
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in today_trades)
            avg_win = (sum(t.pnl for t in winning_trades) / len(winning_trades)) if winning_trades else 0
            avg_loss = (sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else 0
            profit_factor = (-sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
            
            # Generate report
            report = {
                "date": today.isoformat(),
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
                "trades": [self._format_trade(t) for t in today_trades]
            }
            
            # Save report to file
            self._save_report(report, f"daily_report_{today}.json")
            
            # Log summary
            self.logger.info(f"Generated daily report for {today}")
            self.logger.info(f"  Total Trades: {total_trades}")
            self.logger.info(f"  Win Rate: {win_rate:.2f}%")
            self.logger.info(f"  Total P&L: {total_pnl:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a performance report for the last N days.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            dict: Dictionary containing the performance report
        """
        try:
            end_date = datetime.now(self.ist_tz).date()
            start_date = end_date - timedelta(days=days)
            
            # Get trades in date range
            trades = [
                t for t in self.portfolio.get_all_trades()
                if t.exit_time and start_date <= t.exit_time.date() <= end_date
            ]
            
            if not trades:
                return {"message": f"No trades found between {start_date} and {end_date}"}
            
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in trades)
            avg_win = (sum(t.pnl for t in winning_trades) / len(winning_trades)) if winning_trades else 0
            avg_loss = (sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else 0
            profit_factor = (-sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
            
            # Daily P&L
            daily_pnl = {}
            current_date = start_date
            while current_date <= end_date:
                daily_trades = [t for t in trades if t.exit_time.date() == current_date]
                daily_pnl[current_date.isoformat()] = sum(t.pnl for t in daily_trades)
                current_date += timedelta(days=1)
            
            # Generate report
            report = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
                "daily_pnl": daily_pnl,
                "trades": [self._format_trade(t) for t in trades]
            }
            
            # Save report to file
            self._save_report(report, f"performance_report_{start_date}_to_{end_date}.json")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _format_trade(self, trade: Trade) -> Dict[str, Any]:
        """Format a trade for reporting."""
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": "LONG" if trade.quantity > 0 else "SHORT",
            "quantity": abs(trade.quantity),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "pnl": trade.pnl,
            "pnl_percent": trade.pnl_percent,
            "holding_period_minutes": (trade.exit_time - trade.entry_time).total_seconds() / 60 if trade.exit_time else None,
            "exit_reason": trade.exit_reason
        }
    
    def _save_report(self, report: Dict[str, Any], filename: str) -> None:
        """Save a report to a JSON file.
        
        Args:
            report: Report data to save
            filename: Name of the file to save to
        """
        try:
            filepath = self.reports_dir / filename
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Saved report to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving report to {filename}: {str(e)}")
    
    def get_open_positions_summary(self) -> Dict[str, Any]:
        """Get a summary of open positions."""
        try:
            positions = self.portfolio.get_open_positions()
            return {
                "timestamp": datetime.now(self.ist_tz).isoformat(),
                "open_positions": len(positions),
                "total_unrealized_pnl": sum(p.calculate_pnl() for p in positions),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.calculate_pnl(),
                        "age_minutes": (datetime.now(self.ist_tz) - p.entry_time).total_seconds() / 60,
                        "stop_loss": p.stop_loss,
                        "target_price": p.target_price
                    }
                    for p in positions
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting open positions summary: {str(e)}")
            return {"error": str(e)}

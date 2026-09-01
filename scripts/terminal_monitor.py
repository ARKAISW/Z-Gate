"""terminal_monitor.py — Real-time Quant Terminal CLI Dashboard.

Run in any terminal / CMD / PowerShell:
    python scripts/terminal_monitor.py

Features:
  - Rich ASCII live updating quant interface (refreshes every 2s)
  - Live account equity, cash, and buying power metrics
  - Real-time pair signals, Kalman betas, z-scores, and vol regimes
  - Active open trades & options contracts / spot legs
  - Live audit log of Risk Agent decisions
  - Order execution ledger with Alpaca IDs
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure UTF-8 stdout on Windows CMD / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from src.broker import create_broker

DB_PATH = Path("data/trades.db")
console = Console(force_terminal=True, highlight=False)


def get_db_connection() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def fetch_account_data(broker) -> dict[str, Any]:
    try:
        acct = broker.get_account()
        positions = broker.get_positions()
        unrealized_total = sum(getattr(p, "unrealized_pnl", 0.0) or 0.0 for p in positions)
        return {
            "equity": acct.equity,
            "cash": acct.cash,
            "buying_power": acct.buying_power,
            "positions_count": len(positions),
            "positions": positions,
            "unrealized_pnl": unrealized_total,
            "connected": True,
        }
    except Exception as exc:
        return {
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 400000.0,
            "positions_count": 0,
            "positions": [],
            "unrealized_pnl": 0.0,
            "connected": False,
            "error": str(exc),
        }


def format_z_score(z: float | None) -> Text:
    if z is None:
        return Text("-", style="dim")
    if z < -1.5:
        return Text(f"{z:+.2f} (LONG +)", style="bold green")
    if z > 1.5:
        return Text(f"{z:+.2f} (SHORT -)", style="bold red")
    if z < -0.8:
        return Text(f"{z:+.2f} (LONG)", style="green")
    if z > 0.8:
        return Text(f"{z:+.2f} (SHORT)", style="red")
    return Text(f"{z:+.2f}", style="cyan")


def generate_header() -> Panel:
    title = Text()
    title.append(">> ALPACA STAT-ARB QUANT TERMINAL ", style="bold bright_cyan")
    title.append("| ", style="bright_white")
    title.append("HYBRID MULTI-AGENT SYSTEM", style="bold yellow")
    title.append(" | ", style="bright_white")
    title.append("PAPER 24/7", style="bold green on black")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subtitle = Text(f"Deterministic Kalman/OU Engine + Risk Gate | System Time: {now_utc}", style="dim")

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(title, Text("[REFRESH: 2s]", style="dim bright_green"))
    grid.add_row(subtitle, Text("Featherless / Groq Fallback Active", style="dim italic"))

    return Panel(grid, style="bright_blue", box=box.HEAVY)


def generate_kpis(account: dict[str, Any], conn: sqlite3.Connection | None) -> Table:
    kpi_table = Table.grid(expand=True, padding=(0, 2))
    for _ in range(5):
        kpi_table.add_column(justify="center", ratio=1)

    total_trades = 0
    open_trades_count = account.get("positions_count", 0)

    if conn:
        try:
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        except Exception:
            pass

    eq = account.get("equity", 100000.0)
    cash = account.get("cash", 100000.0)
    bp = account.get("buying_power", 400000.0)
    unrealized_pnl = account.get("unrealized_pnl", 0.0)

    # Real P/L calculated directly from Alpaca equity vs $100k starting base
    starting_equity = float(os.getenv("INITIAL_EQUITY", "100000.0"))
    total_real_pnl = eq - starting_equity
    total_real_pct = (total_real_pnl / starting_equity) * 100.0

    pnl_style = "bold green" if total_real_pnl >= 0 else "bold red"
    pnl_str = f"+${total_real_pnl:,.2f} (+{total_real_pct:.2f}%)" if total_real_pnl >= 0 else f"-${abs(total_real_pnl):,.2f} ({total_real_pct:.2f}%)"

    unreal_style = "green" if unrealized_pnl >= 0 else "red"
    unreal_str = f"+${unrealized_pnl:,.2f}" if unrealized_pnl >= 0 else f"-${abs(unrealized_pnl):,.2f}"

    p1 = Panel(f"[bold bright_white]${eq:,.2f}[/]\n[dim]Alpaca Portfolio[/]", title="TOTAL EQUITY", border_style="cyan", box=box.ROUNDED)
    p2 = Panel(f"[bold bright_white]${cash:,.2f}[/]\n[dim]Available Cash[/]", title="CASH BALANCE", border_style="cyan", box=box.ROUNDED)
    p3 = Panel(f"[bold bright_white]${bp:,.2f}[/]\n[dim]Margin Power[/]", title="BUYING POWER", border_style="blue", box=box.ROUNDED)
    p4 = Panel(f"[{pnl_style}]{pnl_str}[/]\n[dim]Unrealized: [{unreal_style}]{unreal_str}[/][/]", title="REAL ACCOUNT P/L", border_style="green" if total_real_pnl >= 0 else "red", box=box.ROUNDED)
    p5 = Panel(f"[bold bright_yellow]{open_trades_count}[/] [dim]Positions[/] / [bold]{total_trades}[/] [dim]Trades[/]\n[dim]Active vs Total[/]", title="PORTFOLIO TRADES", border_style="magenta", box=box.ROUNDED)

    kpi_table.add_row(p1, p2, p3, p4, p5)
    return kpi_table


def generate_pairs_table(conn: sqlite3.Connection | None) -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=False, header_style="bold bright_cyan")
    table.add_column("Pair", style="bold white", width=18)
    table.add_column("Module", style="dim", width=8)
    table.add_column("Direction", justify="center", width=10)
    table.add_column("Z-Score", justify="right", width=16)
    table.add_column("Kalman β", justify="right", width=10)
    table.add_column("Spread", justify="right", width=12)
    table.add_column("Vol Regime", justify="center", width=12)
    table.add_column("Coint p-val", justify="right", width=12)
    table.add_column("Last Signal", justify="right", width=12)

    if not conn:
        table.add_row("Database loading...", "", "", "", "", "", "", "", "")
        return Panel(table, title="[bold bright_white]📊 LIVE STAT-ARB PAIRS & SIGNAL ENGINE[/]", border_style="cyan", box=box.ROUNDED)

    try:
        # Get latest signal for each pair
        query = """
            SELECT s.* FROM signals s
            INNER JOIN (
                SELECT pair_id, MAX(generated_at) as max_generated
                FROM signals GROUP BY pair_id
            ) latest ON s.pair_id = latest.pair_id AND s.generated_at = latest.max_generated
            ORDER BY s.module DESC, s.pair_id ASC
        """
        rows = conn.execute(query).fetchall()
        for r in rows:
            pair = r["pair_id"]
            mod = r["module"]
            direction = r["direction"]
            z = r["z_score"]
            beta = r["beta"]
            p_val = r["coint_pvalue"]
            vol_regime = r["vol_regime"]
            created = r["generated_at"] or ""
            
            spread_val = 0.0
            try:
                rat = json.loads(r["signal_rationale"]) if r["signal_rationale"] else {}
                spread_val = rat.get("current_spread", 0.0)
            except Exception:
                pass

            dir_text = Text(direction.upper(), style="bold green" if direction == "long" else ("bold red" if direction == "short" else "dim"))
            z_text = format_z_score(z)
            reg_style = "bold red" if vol_regime == "EXTREME" else ("yellow" if vol_regime == "HIGH" else "green")
            time_short = created.split("T")[-1][:8] if "T" in created else created[-8:]

            table.add_row(
                pair,
                mod,
                dir_text,
                z_text,
                f"{beta:.4f}" if beta is not None else "-",
                f"{spread_val:+.4f}",
                Text(vol_regime or "NORMAL", style=reg_style),
                f"{p_val:.4f}" if p_val is not None else "-",
                time_short,
            )
    except Exception as exc:
        table.add_row(f"Error: {exc}", "", "", "", "", "", "", "", "")

    return Panel(table, title="[bold bright_white]📊 LIVE STAT-ARB PAIRS & SIGNAL ENGINE[/]", border_style="cyan", box=box.ROUNDED)


def generate_active_trades_table(conn: sqlite3.Connection | None) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_lines=False, header_style="bold bright_yellow")
    table.add_column("Trade ID", style="dim", width=10)
    table.add_column("Pair", style="bold white", width=18)
    table.add_column("Mod", style="dim", width=8)
    table.add_column("Dir", justify="center", width=8)
    table.add_column("Entry Z", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Opened At", justify="right", width=12)

    if not conn:
        return Panel(table, title="[bold bright_white]💼 ACTIVE OPEN POSITIONS[/]", border_style="yellow", box=box.ROUNDED)

    try:
        rows = conn.execute("SELECT * FROM trades WHERE status='open' ORDER BY rowid DESC LIMIT 8").fetchall()
        if not rows:
            table.add_row("No open positions (flat book)", "", "", "", "", "", "")
        for r in rows:
            tid = (r["id"] or "")[:8]
            pair = r["pair_id"]
            mod = r["module"]
            direction = r["direction"]
            entry_z = r["entry_z"]
            status = r["status"]
            opened = r["entry_time"] or ""
            time_short = opened.split("T")[-1][:8] if "T" in opened else opened[-8:]

            dir_text = Text(direction.upper(), style="bold green" if direction == "long" else "bold red")
            stat_text = Text(status.upper(), style="bold green on black")

            table.add_row(
                tid,
                pair,
                mod,
                dir_text,
                f"{entry_z:.2f}" if entry_z is not None else "-",
                stat_text,
                time_short,
            )
    except Exception as exc:
        table.add_row(f"Error: {exc}", "", "", "", "", "", "")

    return Panel(table, title="[bold bright_white]💼 ACTIVE OPEN POSITIONS[/]", border_style="yellow", box=box.ROUNDED)


def generate_recent_orders_table(conn: sqlite3.Connection | None) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_lines=False, header_style="bold bright_green")
    table.add_column("Time", style="dim", width=10)
    table.add_column("Pair / Symbol", style="bold white", width=22)
    table.add_column("Side", justify="center", width=6)
    table.add_column("Qty", justify="right", width=12)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Alpaca Order ID", style="dim", width=12)

    if not conn:
        return Panel(table, title="[bold bright_white]📝 ORDER EXECUTION LEDGER[/]", border_style="green", box=box.ROUNDED)

    try:
        rows = conn.execute("SELECT * FROM orders ORDER BY rowid DESC LIMIT 6").fetchall()
        if not rows:
            table.add_row("No orders executed yet", "", "", "", "", "")
        for r in rows:
            sym = r["symbol"]
            side = (r["side"] or "").upper()
            qty = r["qty"]
            status = r["status"] or "new"
            alpaca_id = (r["alpaca_order_id"] or "")[:8]
            ts = r["created_at"] or ""
            time_short = ts.split("T")[-1][:8] if "T" in ts else ts[-8:]

            side_style = "bold green" if side == "BUY" else "bold red"
            stat_style = "bold bright_green" if status in ("filled", "held") else ("yellow" if status in ("new", "pending_new", "accepted") else "dim")

            table.add_row(
                time_short,
                sym,
                Text(side, style=side_style),
                f"{qty:.4f}" if qty is not None else "-",
                Text(status.upper(), style=stat_style),
                alpaca_id,
            )
    except Exception as exc:
        table.add_row(f"Error: {exc}", "", "", "", "", "")

    return Panel(table, title="[bold bright_white]📝 ORDER EXECUTION LEDGER[/]", border_style="green", box=box.ROUNDED)


def generate_audit_decisions_table(conn: sqlite3.Connection | None) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_lines=False, header_style="bold magenta")
    table.add_column("Time", style="dim", width=10)
    table.add_column("Pair", style="bold white", width=18)
    table.add_column("Decision", justify="center", width=10)
    table.add_column("Rule / Reason", style="dim", width=36)

    if not conn:
        return Panel(table, title="[bold bright_white]🛡️ DETERMINISTIC RISK AGENT AUDIT TRAIL[/]", border_style="magenta", box=box.ROUNDED)

    try:
        rows = conn.execute("SELECT * FROM risk_decisions ORDER BY rowid DESC LIMIT 6").fetchall()
        if not rows:
            table.add_row("No decisions logged", "", "", "")
        for r in rows:
            pair = r["pair_id"]
            approved = bool(r["approved"])
            rule = r["rejection_rule"] or "all_rules_passed"
            reason = r["rejection_reason"] or "Approved for execution"
            ts = r["checked_at"] or ""
            time_short = ts.split("T")[-1][:8] if "T" in ts else ts[-8:]

            dec_text = Text("APPROVED ✅", style="bold green") if approved else Text("REJECTED ❌", style="bold red")
            summary = f"[{rule}] {reason}" if not approved else "All 9 deterministic checks passed"

            table.add_row(
                time_short,
                pair,
                dec_text,
                Text(summary[:36], style="dim" if approved else "yellow"),
            )
    except Exception as exc:
        table.add_row(f"Error: {exc}", "", "", "")

    return Panel(table, title="[bold bright_white]🛡️ DETERMINISTIC RISK AGENT AUDIT TRAIL[/]", border_style="magenta", box=box.ROUNDED)


def make_layout(account_data: dict[str, Any], conn: sqlite3.Connection | None) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="kpis", size=5),
        Layout(name="pairs", size=15),
        Layout(name="bottom", size=10),
    )

    layout["header"].update(generate_header())
    layout["kpis"].update(generate_kpis(account_data, conn))
    layout["pairs"].update(generate_pairs_table(conn))

    layout["bottom"].split_row(
        Layout(generate_active_trades_table(conn), ratio=1),
        Layout(generate_recent_orders_table(conn), ratio=1),
        Layout(generate_audit_decisions_table(conn), ratio=1),
    )

    return layout


def main() -> None:
    broker = create_broker()
    console.print("[bold green]Starting Alpaca Stat-Arb Quant Terminal Monitor...[/]")
    time.sleep(0.5)

    with Live(console=console, screen=True, refresh_per_second=1) as live:
        while True:
            try:
                conn = get_db_connection()
                account_data = fetch_account_data(broker)
                live.update(make_layout(account_data, conn))
                if conn:
                    conn.close()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                console.print(f"[bold red]Monitor error:[/] {exc}")
            time.sleep(2)


if __name__ == "__main__":
    main()

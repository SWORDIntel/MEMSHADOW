#!/usr/bin/env python3
"""
MEMSHADOW Interactive TUI Dashboard
Classification: UNCLASSIFIED
"""

import asyncio
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Error: 'rich' library not installed.")
    print("Install with: pip install rich")
    sys.exit(1)

console = Console()


class MEMSHADOWDashboard:
    """MEMSHADOW Interactive TUI Dashboard"""

    def __init__(self):
        self.compose_file = "/opt/memshadow/docker-compose.hardened.yml"
        self.running = True
        self.last_update = datetime.now()
        self.log_lines: List[str] = []
        self.max_log_lines = 20

    def run_command(self, cmd: List[str], timeout: int = 5) -> tuple[str, int]:
        """Run shell command and return output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "Command timeout", 1
        except Exception as e:
            return f"Error: {str(e)}", 1

    def get_docker_status(self) -> Dict[str, str]:
        """Get Docker container status"""
        cmd = ["docker", "ps", "--format", "{{.Names}}|{{.Status}}|{{.State}}"]
        output, code = self.run_command(cmd)

        status = {}
        if code == 0 and output:
            for line in output.split('\n'):
                if 'memshadow' in line.lower():
                    parts = line.split('|')
                    if len(parts) >= 3:
                        name = parts[0].replace('memshadow-', '')
                        state = parts[2]
                        status[name] = state

        return status

    def check_health_endpoint(self, url: str = "http://localhost:8000/api/v1/health") -> bool:
        """Check if health endpoint is responding"""
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            return True
        except:
            return False

    def get_system_metrics(self) -> Dict[str, str]:
        """Get system metrics"""
        metrics = {}

        # CPU usage
        cmd = ["top", "-bn1"]
        output, code = self.run_command(cmd, timeout=2)
        if code == 0:
            for line in output.split('\n'):
                if 'Cpu(s)' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'id,' in part:
                            idle = float(part.replace('id,', ''))
                            metrics['cpu'] = f"{100 - idle:.1f}%"
                            break

        # Memory usage
        cmd = ["free", "-h"]
        output, code = self.run_command(cmd, timeout=2)
        if code == 0:
            lines = output.split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 7:
                    metrics['memory'] = f"{parts[2]}/{parts[1]}"

        # Disk usage
        cmd = ["df", "-h", "/opt/memshadow"]
        output, code = self.run_command(cmd, timeout=2)
        if code == 0:
            lines = output.split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    metrics['disk'] = f"{parts[2]}/{parts[1]} ({parts[4]})"

        return metrics

    def get_recent_logs(self, service: str = "memshadow", lines: int = 20) -> List[str]:
        """Get recent log lines"""
        cmd = [
            "docker", "logs", f"memshadow-{service}",
            "--tail", str(lines)
        ]
        output, code = self.run_command(cmd, timeout=3)

        if code == 0 and output:
            return output.split('\n')[-lines:]
        return []

    def create_header(self) -> Panel:
        """Create dashboard header"""
        header_text = Text()
        header_text.append("MEMSHADOW", style="bold red")
        header_text.append(" | ", style="white")
        header_text.append("Advanced Offensive Security Platform", style="bold cyan")
        header_text.append("\n")
        header_text.append(f"Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")

        return Panel(
            Align.center(header_text),
            style="bold white on black",
            box=box.DOUBLE
        )

    def create_service_status(self) -> Panel:
        """Create service status panel"""
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Service", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("State", justify="center")

        docker_status = self.get_docker_status()

        services = ["memshadow", "postgres", "redis", "chromadb", "waf", "ids"]

        for service in services:
            state = docker_status.get(service, "unknown")

            if state == "running":
                status_icon = "●"
                status_style = "bold green"
                state_text = "Running"
            elif state == "exited":
                status_icon = "○"
                status_style = "bold red"
                state_text = "Stopped"
            else:
                status_icon = "◐"
                status_style = "bold yellow"
                state_text = "Unknown"

            table.add_row(
                service.upper(),
                Text(status_icon, style=status_style),
                Text(state_text, style=status_style)
            )

        return Panel(table, title="[bold]Service Status[/bold]", border_style="green")

    def create_health_check(self) -> Panel:
        """Create health check panel"""
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")

        # API Health
        api_healthy = self.check_health_endpoint()
        table.add_row(
            "API Endpoint",
            Text("✓ Healthy" if api_healthy else "✗ Down",
                 style="bold green" if api_healthy else "bold red")
        )

        # Docker Daemon
        _, docker_code = self.run_command(["docker", "info"], timeout=2)
        table.add_row(
            "Docker Daemon",
            Text("✓ Running" if docker_code == 0 else "✗ Not Running",
                 style="bold green" if docker_code == 0 else "bold red")
        )

        # PostgreSQL
        pg_cmd = ["docker", "exec", "memshadow-postgres", "pg_isready", "-U", "memshadow"]
        _, pg_code = self.run_command(pg_cmd, timeout=2)
        table.add_row(
            "PostgreSQL",
            Text("✓ Ready" if pg_code == 0 else "✗ Not Ready",
                 style="bold green" if pg_code == 0 else "bold red")
        )

        # Redis
        redis_cmd = ["docker", "exec", "memshadow-redis", "redis-cli", "ping"]
        redis_out, redis_code = self.run_command(redis_cmd, timeout=2)
        redis_ok = redis_code == 0 and "PONG" in redis_out
        table.add_row(
            "Redis",
            Text("✓ Responding" if redis_ok else "✗ Not Responding",
                 style="bold green" if redis_ok else "bold red")
        )

        return Panel(table, title="[bold]Health Checks[/bold]", border_style="blue")

    def create_metrics(self) -> Panel:
        """Create system metrics panel"""
        metrics = self.get_system_metrics()

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        table.add_row("CPU Usage", metrics.get('cpu', 'N/A'))
        table.add_row("Memory", metrics.get('memory', 'N/A'))
        table.add_row("Disk", metrics.get('disk', 'N/A'))

        return Panel(table, title="[bold]System Metrics[/bold]", border_style="yellow")

    def create_logs(self) -> Panel:
        """Create log viewer panel"""
        logs = self.get_recent_logs("memshadow", self.max_log_lines)

        if not logs:
            log_text = Text("No logs available", style="dim")
        else:
            log_text = Text()
            for log_line in logs[-self.max_log_lines:]:
                if "ERROR" in log_line:
                    log_text.append(log_line + "\n", style="bold red")
                elif "WARN" in log_line:
                    log_text.append(log_line + "\n", style="bold yellow")
                elif "INFO" in log_line:
                    log_text.append(log_line + "\n", style="green")
                else:
                    log_text.append(log_line + "\n", style="dim")

        return Panel(
            log_text,
            title="[bold]Recent Logs (memshadow)[/bold]",
            border_style="magenta",
            height=15
        )

    def create_quick_actions(self) -> Panel:
        """Create quick actions panel"""
        actions = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        actions.add_column("Key", style="bold cyan")
        actions.add_column("Action", style="white")

        actions.add_row("q", "Quit")
        actions.add_row("r", "Refresh")
        actions.add_row("s", "Start Services")
        actions.add_row("x", "Stop Services")
        actions.add_row("h", "Show Help")

        return Panel(
            actions,
            title="[bold]Quick Actions[/bold]",
            border_style="cyan"
        )

    def create_layout(self) -> Layout:
        """Create dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="services", size=13),
            Layout(name="health", size=10),
        )

        layout["right"].split_column(
            Layout(name="metrics", size=9),
            Layout(name="actions", size=14)
        )

        layout["header"].update(self.create_header())
        layout["services"].update(self.create_service_status())
        layout["health"].update(self.create_health_check())
        layout["metrics"].update(self.create_metrics())
        layout["actions"].update(self.create_quick_actions())
        layout["footer"].update(self.create_logs())

        return layout

    def run_interactive(self):
        """Run interactive TUI dashboard"""
        console.clear()

        try:
            with Live(self.create_layout(), console=console, refresh_per_second=1) as live:
                while self.running:
                    time.sleep(2)  # Update every 2 seconds
                    self.last_update = datetime.now()
                    live.update(self.create_layout())
        except KeyboardInterrupt:
            console.clear()
            console.print("\n[bold red]Dashboard stopped.[/bold red]")

    def run_once(self):
        """Run dashboard once (non-interactive)"""
        console.clear()
        console.print(self.create_layout())


def main():
    """Main entry point"""
    try:
        dashboard = MEMSHADOWDashboard()

        # Check if we can run interactive mode
        if sys.stdout.isatty():
            console.print("[bold cyan]Starting MEMSHADOW Interactive Dashboard...[/bold cyan]")
            console.print("[dim]Press Ctrl+C to exit[/dim]\n")
            time.sleep(1)
            dashboard.run_interactive()
        else:
            # Non-interactive mode
            dashboard.run_once()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TEMPEST Terminal User Interface
Military-grade interactive interface for MEMSHADOW operations

CLASSIFICATION: UNCLASSIFIED
For authorized security research and defensive analysis only.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
    from rich.tree import Tree
    from rich import box
    from rich.text import Text
except ImportError:
    print("[!] Error: 'rich' library not installed")
    print("[!] Install with: pip install rich")
    sys.exit(1)

import structlog

logger = structlog.get_logger()


class TempestTUI:
    """
    TEMPEST-style Terminal User Interface for MEMSHADOW
    """

    def __init__(self):
        self.console = Console()
        self.missions_dir = Path("missions")
        self.current_mission = None
        self.classification = "UNCLASSIFIED"

    def display_banner(self):
        """Display TEMPEST military-style banner"""
        # Classification header
        classification_header = "[bold green on black]" + "=" * 80 + "\n" + \
                               self.classification.center(80) + "\n" + \
                               "=" * 80 + "[/bold green on black]"
        self.console.print(classification_header)

        banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   [bold bright_green]███╗   ███╗███████╗███╗   ███╗███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗[/bold bright_green]  ║
║   [bold bright_green]████╗ ████║██╔════╝████╗ ████║██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║[/bold bright_green]  ║
║   [bold bright_green]██╔████╔██║█████╗  ██╔████╔██║███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║[/bold bright_green]  ║
║   [bold bright_green]██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║[/bold bright_green]  ║
║   [bold bright_green]██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝[/bold bright_green]  ║
║   [bold bright_green]╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝[/bold bright_green]   ║
║                                                                           ║
║            [bold cyan]TEMPEST Class C Military Operations Interface[/bold cyan]              ║
║                    [yellow]Version 2.0 - Integration Build[/yellow]                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
        self.console.print(banner)

        # System info
        zulu_time = datetime.now(timezone.utc).strftime("%Y%m%d %H%M%SZ")
        info_panel = Panel(
            f"[cyan]System Status:[/cyan] OPERATIONAL\n"
            f"[cyan]Time (Zulu):[/cyan] {zulu_time}\n"
            f"[cyan]Classification:[/cyan] {self.classification}\n"
            f"[cyan]Operator:[/cyan] Authorized User",
            title="[bold green]SYSTEM INFORMATION[/bold green]",
            border_style="green"
        )
        self.console.print(info_panel)
        self.console.print()

    def display_main_menu(self) -> str:
        """Display main menu and get user selection"""
        menu_table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
        menu_table.add_column("Option", style="bold yellow", width=10)
        menu_table.add_column("Description", style="white")

        menu_table.add_row("[1]", "Mission Operations (ENUMERATE > PLAN > EXECUTE)")
        menu_table.add_row("[2]", "Agent Status & Monitoring")
        menu_table.add_row("[3]", "Intelligence Dashboard (IoCs & Vulnerabilities)")
        menu_table.add_row("[4]", "Workflow Engine (Automated Operations)")
        menu_table.add_row("[5]", "System Configuration")
        menu_table.add_row("[Q]", "Exit TEMPEST Interface")

        panel = Panel(
            menu_table,
            title="[bold green]═══ TEMPEST MAIN MENU ═══[/bold green]",
            border_style="green"
        )
        self.console.print(panel)

        choice = Prompt.ask(
            "[bold cyan]Select option[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "q", "Q"],
            default="1"
        )

        return choice.upper()

    def mission_operations_menu(self):
        """Mission operations interface"""
        while True:
            self.console.clear()
            self.display_banner()

            # Mission selection
            missions = self.discover_missions()

            mission_table = Table(
                title="[bold green]Available Missions[/bold green]",
                box=box.ROUNDED,
                border_style="cyan"
            )
            mission_table.add_column("#", style="yellow", width=5)
            mission_table.add_column("Mission ID", style="cyan")
            mission_table.add_column("Name", style="white")
            mission_table.add_column("Classification", style="bold")

            for idx, mission in enumerate(missions, 1):
                classification_color = self.get_classification_color(mission.get("classification", "UNCLASSIFIED"))
                mission_table.add_row(
                    str(idx),
                    mission.get("mission_id", "unknown"),
                    mission.get("name", "Unknown Mission"),
                    f"[{classification_color}]{mission.get('classification', 'UNCLASSIFIED')}[/{classification_color}]"
                )

            self.console.print(mission_table)
            self.console.print()

            choice = Prompt.ask(
                "[bold cyan]Select mission number (or 'b' to go back)[/bold cyan]",
                default="b"
            )

            if choice.lower() == 'b':
                break

            try:
                mission_idx = int(choice) - 1
                if 0 <= mission_idx < len(missions):
                    selected_mission = missions[mission_idx]
                    self.execute_mission_workflow(selected_mission)
                else:
                    self.console.print("[red]Invalid mission number[/red]")
                    self.console.input("\nPress Enter to continue...")
            except ValueError:
                self.console.print("[red]Invalid input[/red]")
                self.console.input("\nPress Enter to continue...")

    def agent_monitoring_menu(self):
        """Agent monitoring interface"""
        self.console.clear()
        self.display_banner()

        agent_table = Table(
            title="[bold green]SWARM Agent Status[/bold green]",
            box=box.ROUNDED,
            border_style="cyan"
        )
        agent_table.add_column("Agent Type", style="yellow")
        agent_table.add_column("Status", style="white")
        agent_table.add_column("Pending Tasks", style="cyan")
        agent_table.add_column("Completed", style="green")

        agents = [
            ("recon", "ACTIVE", "3", "127"),
            ("crawler", "ACTIVE", "1", "45"),
            ("apimapper", "ACTIVE", "0", "23"),
            ("authtest", "ACTIVE", "2", "89"),
            ("bgp", "STANDBY", "0", "12"),
            ("blockchain", "STANDBY", "0", "8"),
            ("wifi", "ACTIVE", "1", "5"),
            ("webscan", "ACTIVE", "4", "34")
        ]

        for agent_type, status, pending, completed in agents:
            status_color = "green" if status == "ACTIVE" else "yellow"
            agent_table.add_row(
                agent_type,
                f"[{status_color}]{status}[/{status_color}]",
                pending,
                completed
            )

        self.console.print(agent_table)
        self.console.print()
        self.console.input("Press Enter to return to main menu...")

    def intelligence_dashboard_menu(self):
        """Intelligence dashboard interface"""
        self.console.clear()
        self.display_banner()

        # IoCs Table
        ioc_table = Table(
            title="[bold green]Recent Indicators of Compromise[/bold green]",
            box=box.ROUNDED,
            border_style="cyan"
        )
        ioc_table.add_column("Type", style="yellow")
        ioc_table.add_column("Value", style="white")
        ioc_table.add_column("Threat Level", style="bold")
        ioc_table.add_column("Source", style="cyan")

        # Sample IoCs
        iocs = [
            ("IPv4", "192.168.1.100", "HIGH", "recon_agent"),
            ("Domain", "malicious.example.com", "CRITICAL", "webscan_agent"),
            ("MD5", "d41d8cd98f00b204e9800998ecf8427e", "MEDIUM", "crawler_agent"),
            ("CVE", "CVE-2025-64446", "CRITICAL", "webscan_agent")
        ]

        for ioc_type, value, threat, source in iocs:
            threat_color = self.get_threat_color(threat)
            ioc_table.add_row(
                ioc_type,
                value,
                f"[{threat_color}]{threat}[/{threat_color}]",
                source
            )

        self.console.print(ioc_table)
        self.console.print()

        # Vulnerabilities Table
        vuln_table = Table(
            title="[bold green]Discovered Vulnerabilities[/bold green]",
            box=box.ROUNDED,
            border_style="cyan"
        )
        vuln_table.add_column("Type", style="yellow")
        vuln_table.add_column("CVSS", style="red")
        vuln_table.add_column("Target", style="white")
        vuln_table.add_column("Risk", style="bold")

        vulns = [
            ("SQLi", "9.8", "target.example.com/login", "CRITICAL"),
            ("XSS", "7.5", "target.example.com/search", "HIGH"),
            ("SSRF", "8.0", "target.example.com/api", "HIGH")
        ]

        for vuln_type, cvss, target, risk in vulns:
            risk_color = self.get_threat_color(risk)
            vuln_table.add_row(
                vuln_type,
                cvss,
                target,
                f"[{risk_color}]{risk}[/{risk_color}]"
            )

        self.console.print(vuln_table)
        self.console.print()
        self.console.input("Press Enter to return to main menu...")

    def workflow_engine_menu(self):
        """Workflow engine interface"""
        self.console.clear()
        self.display_banner()

        workflow_panel = Panel(
            "[bold cyan]ENUMERATE > PLAN > EXECUTE Workflow[/bold cyan]\n\n"
            "This automated workflow performs:\n"
            "  [green]1. ENUMERATE[/green] - Network discovery, service detection\n"
            "  [yellow]2. PLAN[/yellow] - Attack chain analysis, target prioritization\n"
            "  [red]3. EXECUTE[/red] - Automated exploitation (requires confirmation)\n\n"
            "[bold yellow]⚠ WARNING:[/bold yellow] EXECUTE phase requires explicit authorization",
            title="[bold green]Workflow Engine[/bold green]",
            border_style="green"
        )
        self.console.print(workflow_panel)
        self.console.print()

        target = Prompt.ask("[cyan]Enter target (CIDR or IP)[/cyan]", default="192.168.1.0/24")

        auto_execute = Confirm.ask(
            "[yellow]⚠ Enable auto-execute? (DANGEROUS - requires authorization)[/yellow]",
            default=False
        )

        if Confirm.ask(f"\n[bold]Start workflow against {target}?[/bold]", default=True):
            self.run_workflow_simulation(target, auto_execute)

        self.console.input("\nPress Enter to return to main menu...")

    def run_workflow_simulation(self, target: str, auto_execute: bool):
        """Simulate workflow execution with progress tracking"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:

            # ENUMERATE phase
            enumerate_task = progress.add_task(
                "[green]ENUMERATE: Network discovery...",
                total=100
            )
            for i in range(100):
                progress.update(enumerate_task, advance=1)
                asyncio.run(asyncio.sleep(0.02))

            self.console.print("[green]✓ ENUMERATE complete - 15 hosts discovered[/green]")

            # PLAN phase
            plan_task = progress.add_task(
                "[yellow]PLAN: Analyzing attack chains...",
                total=100
            )
            for i in range(100):
                progress.update(plan_task, advance=1)
                asyncio.run(asyncio.sleep(0.015))

            self.console.print("[yellow]✓ PLAN complete - 8 attack chains identified[/yellow]")

            # EXECUTE phase
            if auto_execute:
                execute_task = progress.add_task(
                    "[red]EXECUTE: Running exploits...",
                    total=100
                )
                for i in range(100):
                    progress.update(execute_task, advance=1)
                    asyncio.run(asyncio.sleep(0.01))

                self.console.print("[red]✓ EXECUTE complete - 3 successful compromises[/red]")
            else:
                self.console.print("[yellow]⊘ EXECUTE skipped (manual confirmation required)[/yellow]")

    def execute_mission_workflow(self, mission: Dict[str, Any]):
        """Execute mission with real-time monitoring"""
        self.console.clear()
        self.display_banner()

        # Mission briefing
        briefing = Panel(
            f"[bold cyan]Mission ID:[/bold cyan] {mission.get('mission_id')}\n"
            f"[bold cyan]Mission Name:[/bold cyan] {mission.get('name')}\n"
            f"[bold cyan]Classification:[/bold cyan] {mission.get('classification')}\n"
            f"[bold cyan]Description:[/bold cyan] {mission.get('description', 'No description')}\n\n"
            f"[yellow]⚠ This operation will deploy SWARM agents for autonomous execution[/yellow]",
            title="[bold green]MISSION BRIEFING[/bold green]",
            border_style="green"
        )
        self.console.print(briefing)
        self.console.print()

        if not Confirm.ask("[bold]Authorize mission execution?[/bold]", default=False):
            self.console.print("[yellow]Mission aborted by operator[/yellow]")
            self.console.input("\nPress Enter to continue...")
            return

        # Simulate mission execution
        self.console.print("\n[bold green]Mission authorized - deploying agents...[/bold green]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            stages = [
                "Initializing SWARM coordinator",
                "Deploying recon agents",
                "Executing network enumeration",
                "Running vulnerability scans",
                "Generating intelligence report"
            ]

            for stage in stages:
                task = progress.add_task(f"[cyan]{stage}...", total=None)
                asyncio.run(asyncio.sleep(1.5))
                progress.remove_task(task)
                self.console.print(f"[green]✓ {stage}[/green]")

        self.console.print("\n[bold green]═══ MISSION COMPLETE ═══[/bold green]\n")
        self.console.input("Press Enter to return to mission menu...")

    def discover_missions(self) -> List[Dict[str, Any]]:
        """Discover available missions"""
        missions = []

        # Check missions/examples/
        examples_dir = self.missions_dir / "examples"
        if examples_dir.exists():
            for mission_file in examples_dir.glob("*.yaml"):
                missions.append({
                    "mission_id": mission_file.stem,
                    "name": mission_file.stem.replace("_", " ").title(),
                    "classification": "UNCLASSIFIED",
                    "path": str(mission_file),
                    "description": "Example mission"
                })

        # Check missions/classified/
        classified_dir = self.missions_dir / "classified"
        if classified_dir.exists():
            for mission_file in classified_dir.glob("*.yaml"):
                # Extract classification from filename
                filename = mission_file.stem
                if filename.startswith("TS_"):
                    classification = "TOP SECRET"
                elif filename.startswith("SECRET_"):
                    classification = "SECRET"
                elif filename.startswith("CONFIDENTIAL_"):
                    classification = "CONFIDENTIAL"
                elif filename.startswith("FOUO_"):
                    classification = "FOR OFFICIAL USE ONLY"
                else:
                    classification = "UNCLASSIFIED"

                missions.append({
                    "mission_id": filename,
                    "name": filename.split("_", 1)[1].replace("_", " ").title() if "_" in filename else filename,
                    "classification": classification,
                    "path": str(mission_file),
                    "description": "Classified mission"
                })

        return missions

    def get_classification_color(self, classification: str) -> str:
        """Get color for classification level"""
        colors = {
            "TOP SECRET": "red",
            "SECRET": "magenta",
            "CONFIDENTIAL": "yellow",
            "FOR OFFICIAL USE ONLY": "cyan",
            "UNCLASSIFIED": "green"
        }
        return colors.get(classification, "white")

    def get_threat_color(self, threat_level: str) -> str:
        """Get color for threat level"""
        colors = {
            "CRITICAL": "red",
            "HIGH": "yellow",
            "MEDIUM": "cyan",
            "LOW": "green"
        }
        return colors.get(threat_level, "white")

    def run(self):
        """Main TUI loop"""
        try:
            while True:
                self.console.clear()
                self.display_banner()

                choice = self.display_main_menu()

                if choice == "1":
                    self.mission_operations_menu()
                elif choice == "2":
                    self.agent_monitoring_menu()
                elif choice == "3":
                    self.intelligence_dashboard_menu()
                elif choice == "4":
                    self.workflow_engine_menu()
                elif choice == "5":
                    self.console.print("[yellow]System configuration - Coming soon[/yellow]")
                    self.console.input("\nPress Enter to continue...")
                elif choice == "Q":
                    if Confirm.ask("\n[yellow]Exit TEMPEST interface?[/yellow]", default=True):
                        break

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Operation interrupted by operator[/yellow]")

        finally:
            # Closing banner
            self.console.print("\n")
            closing = Panel(
                "[bold green]TEMPEST interface terminated[/bold green]\n"
                f"[cyan]Time (Zulu):[/cyan] {datetime.now(timezone.utc).strftime('%Y%m%d %H%M%SZ')}\n"
                "[cyan]Status:[/cyan] SHUTDOWN COMPLETE",
                title="[bold yellow]SYSTEM SHUTDOWN[/bold yellow]",
                border_style="yellow"
            )
            self.console.print(closing)
            self.console.print("\n[bold green]" + "=" * 80 + "[/bold green]")
            self.console.print("[bold green]" + self.classification.center(80) + "[/bold green]")
            self.console.print("[bold green]" + "=" * 80 + "[/bold green]\n")


def main():
    """Entry point for TEMPEST TUI"""
    tui = TempestTUI()
    tui.run()


if __name__ == "__main__":
    main()

from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.containers import Grid
from rich.text import Text
from rich.table import Table
from .core import PowerSteering
from .renderer import ConsoleRenderer


class InfoBar(Static):
    """Info bar showing statistics and notifications"""
    def __init__(self):
        super().__init__("")
        self.stats_text = ""
        self.notification = ""
        self._update_content()

    def set_stats(self, total_cars: int, undriven_cars: int) -> None:
        """Update the statistics display"""
        self.stats_text = f"Cars: {total_cars} | Undriven: {undriven_cars}"
        self._update_content()

    def notify(self, message: str) -> None:
        """Show a notification message"""
        self.notification = message
        self._update_content()

    def clear_notification(self) -> None:
        """Clear the notification message"""
        self.notification = ""
        self._update_content()

    def _update_content(self) -> None:
        """Update the displayed content"""
        content = Text.assemble(
            (self.stats_text, "bold"),
            "  ",
            (self.notification, "italic green")
        )
        self.update(content)

class MainDisplay(Static):
    """Main display area for car information and operations"""
    def __init__(self):
        super().__init__()
        self.ps = None
        self.renderer = ConsoleRenderer(quiet=True)
        self.show_stats = True
        self.show_undriven = False

    def set_powersteering(self, ps: PowerSteering) -> None:
        """Set the PowerSteering instance and update display"""
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        """Update the main display content"""
        if not self.ps:
            self.update("Loading...")
            return

        # Get tables from renderer
        stats_table = self.renderer.create_stats_table(
            self.ps.cars,
            self.ps.undriven_cars,
            self.ps.has_custom_ffb
        )
        undriven_table = self.renderer.create_undriven_table(self.ps.undriven_cars)
        custom_ffb_table = self.renderer.create_custom_ffb_table(
            self.ps.cars,
            self.ps.has_custom_ffb
        )

        # Create a container table to hold visible tables
        container = Table(show_header=False, show_edge=False, padding=1)

        if self.show_stats:
            container.add_row(stats_table)
            container.add_row(custom_ffb_table)

        if self.show_undriven:
            container.add_row(undriven_table)

        self.update(container)

class PowerSteeringApp(App):
    """A Textual app to manage RSF power steering settings"""

    TITLE = "PowerSteering"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "toggle_stats", "Statistics"),
        ("u", "toggle_undriven", "Undriven Cars")
    ]

    CSS = """
    InfoBar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }

    MainDisplay {
        height: 1fr;
        padding: 1;
    }

    Grid {
        layout: grid;
        grid-size: 1;
        grid-rows: 1 1fr auto;
    }
    """

    def __init__(self, rsf_path: str):
        super().__init__()
        self.rsf_path = rsf_path
        self.ps = PowerSteering(rsf_path)

    def compose(self) -> ComposeResult:
        yield Grid(
            InfoBar(),
            MainDisplay(),
            Footer(),
        )

    def on_mount(self) -> None:
        """After the app is mounted, initialize the display"""
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh all display components"""
        info_bar = self.query_one(InfoBar)
        main_display = self.query_one(MainDisplay)

        info_bar.set_stats(len(self.ps.cars), len(self.ps.undriven_cars))
        main_display.set_powersteering(self.ps)

    def action_refresh(self) -> None:
        """Reload PowerSteering data and refresh display"""
        self.ps = PowerSteering(self.rsf_path)
        self._refresh_display()
        self.query_one(InfoBar).notify("Data refreshed")

    def action_toggle_stats(self) -> None:
        """Toggle statistics tables visibility"""
        main_display = self.query_one(MainDisplay)
        main_display.show_stats = not main_display.show_stats
        if main_display.show_stats:
            main_display.show_undriven = False
        main_display._update_display()
        self.query_one(InfoBar).notify("Showing statistics view")

    def action_toggle_undriven(self) -> None:
        """Toggle undriven cars table visibility"""
        main_display = self.query_one(MainDisplay)
        main_display.show_undriven = not main_display.show_undriven
        if main_display.show_undriven:
            main_display.show_stats = False
        main_display._update_display()
        self.query_one(InfoBar).notify("Showing undriven cars view")


from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, DataTable
from textual.containers import Grid, VerticalScroll
from rich.text import Text
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

class StatsView(Static):
    """View for displaying car statistics"""
    def __init__(self):
        super().__init__()
        self.ps = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(
            DataTable(id="stats-table"),
            DataTable(id="ffb-table")
        )

    def set_powersteering(self, ps: PowerSteering) -> None:
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        if not self.ps:
            self.update("Loading...")
            return

        # Update stats table
        stats_table = self.query_one("#stats-table", DataTable)
        stats_table.clear()
        stats_table.add_columns("Metric", "Count")
        
        total_cars = len(self.ps.cars)
        ffb_cars = sum(1 for car in self.ps.cars.values() if self.ps.has_custom_ffb(car))
        undriven_cars_count = len(self.ps.undriven_cars)
        
        stats_table.add_rows([
            ("Total Cars", str(total_cars)),
            ("Cars with Custom FFB", str(ffb_cars)),
            ("Undriven Cars", str(undriven_cars_count))
        ])

        # Update FFB table
        ffb_table = self.query_one("#ffb-table", DataTable)
        ffb_table.clear()
        ffb_table.add_columns("Car", "Weight", "Steering", "Drivetrain", "FFB Settings", "Global FFB")
        
        for car in sorted(self.ps.cars.values(), key=lambda x: x.name):
            if self.ps.has_custom_ffb(car):
                ffb_table.add_row(
                    f"{car.id} - {car.name}",
                    str(car.weight),
                    f"{car.steering_wheel}°",
                    car.drive_train,
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
                )

class ClusterView(Static):
    """View for displaying cluster statistics"""
    def __init__(self):
        super().__init__()
        self.ps = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(
            DataTable(id="cluster-stats-table"),
            DataTable(id="selected-cars-table")
        )

    def set_powersteering(self, ps: PowerSteering) -> None:
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        if not self.ps:
            self.update("Loading...")
            return

        selected = self.ps.select_training_sample(10)
        cluster_data = self.ps.get_cluster_data(selected)

        # Update cluster stats table
        stats_table = self.query_one("#cluster-stats-table", DataTable)
        stats_table.clear()
        stats_table.add_columns("Cluster", "Size", "Avg Weight", "Avg Steering", "Drivetrain Distribution")

        for cluster_id, cars in sorted(cluster_data.items()):
            weights = [car.weight for car in cars]
            steering = [car.steering_wheel for car in cars]
            drive_counts = {}
            for car in cars:
                drive_counts[car.drive_train] = drive_counts.get(car.drive_train, 0) + 1
            drive_dist = ", ".join(f"{dt}: {count}" for dt, count in drive_counts.items())

            stats_table.add_row(
                str(cluster_id),
                str(len(cars)),
                f"{min(weights)}-{max(weights)} kg",
                f"{min(steering)}-{max(steering)}°",
                drive_dist
            )

        # Update selected cars table
        cars_table = self.query_one("#selected-cars-table", DataTable)
        cars_table.clear()
        cars_table.add_columns("Cluster", "Car", "Weight", "Steering", "Drivetrain", "FFB (T/G/S)")

        for car in selected:
            cars_table.add_row(
                str(car.cluster),
                f"{car.id} - {car.name}",
                str(car.weight),
                f"{car.steering_wheel}°",
                car.drive_train,
                f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
            )

class UndrivenView(Static):
    """View for displaying undriven cars"""
    def __init__(self):
        super().__init__()
        self.ps = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="undriven-table")

    def set_powersteering(self, ps: PowerSteering) -> None:
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        if not self.ps:
            self.update("Loading...")
            return

        table = self.query_one("#undriven-table", DataTable)
        table.clear()
        table.add_columns("Car", "Weight", "Steering", "Drivetrain")

        for car in sorted(self.ps.undriven_cars.values(), key=lambda x: x.name):
            table.add_row(
                f"{car.id} - {car.name}",
                str(car.weight),
                f"{car.steering_wheel}°",
                car.drive_train
            )

class MainDisplay(Static):
    """Main display area for car information and operations"""
    def __init__(self):
        super().__init__()
        self.ps = None
        self.stats_view = StatsView()
        self.undriven_view = UndrivenView()
        self.cluster_view = ClusterView()
        self.current_view = self.stats_view

    def compose(self) -> ComposeResult:
        yield self.stats_view
        yield self.undriven_view
        yield self.cluster_view
        self.undriven_view.display = False
        self.cluster_view.display = False

    def set_powersteering(self, ps: PowerSteering) -> None:
        """Set the PowerSteering instance and update display"""
        self.ps = ps
        self.stats_view.set_powersteering(ps)
        self.undriven_view.set_powersteering(ps)
        self.cluster_view.set_powersteering(ps)

    def show_stats(self) -> None:
        """Switch to statistics view"""
        self.stats_view.display = True
        self.undriven_view.display = False
        self.current_view = self.stats_view

    def show_undriven(self) -> None:
        """Switch to undriven cars view"""
        self.stats_view.display = False
        self.undriven_view.display = True
        self.cluster_view.display = False
        self.current_view = self.undriven_view

    def show_clusters(self) -> None:
        """Switch to cluster view"""
        self.stats_view.display = False
        self.undriven_view.display = False
        self.cluster_view.display = True
        self.current_view = self.cluster_view

class PowerSteeringApp(App):
    """A Textual app to manage RSF power steering settings"""

    TITLE = "PowerSteering"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "toggle_stats", "Statistics"),
        ("u", "toggle_undriven", "Undriven Cars"),
        ("c", "toggle_clusters", "Clusters")
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

    DataTable {
        height: 1fr;
        width: 100%;
        border: solid $primary-background;
    }

    DataTable > .datatable--header {
        background: $panel;
        color: $text;
    }

    DataTable > .datatable--body {
        background: $surface;
    }

    VerticalScroll {
        height: 1fr;
        width: 100%;
        border: solid $primary-background;
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
        """Switch to statistics view"""
        main_display = self.query_one(MainDisplay)
        main_display.show_stats()
        self.query_one(InfoBar).notify("Showing statistics view")

    def action_toggle_undriven(self) -> None:
        """Switch to undriven cars view"""
        main_display = self.query_one(MainDisplay)
        main_display.show_undriven()
        self.query_one(InfoBar).notify("Showing undriven cars view")

    def action_toggle_clusters(self) -> None:
        """Switch to cluster view"""
        main_display = self.query_one(MainDisplay)
        main_display.show_clusters()
        self.query_one(InfoBar).notify("Showing cluster statistics")


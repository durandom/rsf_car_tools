from abc import abstractmethod
from typing import List, Optional, Protocol
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, DataTable
from .models import Car
from textual.containers import Grid, VerticalScroll
from rich.text import Text
from .core import PowerSteering
from .renderer import ConsoleRenderer


class BaseCarTableView(Static):
    """Base class for views that display car tables"""

    @abstractmethod
    def _update_display(self) -> None:
        """Update the display with current data"""
        raise NotImplementedError

    def _setup_car_table(self, table_id: str, columns: Optional[List[str]] = None) -> DataTable:
        """Setup a DataTable with given columns"""
        if columns is None:
            columns = ["Id", "Car", "Weight", "Steering", "Drivetrain", "FFB Settings"]
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=True)
        table.add_columns(*columns)
        return table

    def _format_car_row(self, car: Car, include_ffb: bool = True) -> List[str]:
        """Format car data into a row"""
        row = [
            f"{car.id}",
            f"{car.name}",
            str(car.weight),
            f"{car.steering_wheel}°",
            car.drive_train,
            f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
        ]
        return row


class InfoBar(Static):
    """Info bar showing statistics and notifications"""
    def __init__(self):
        super().__init__("")
        self.stats_text = ""
        self.notification = ""
        self._update_content()

    def set_stats(self, ps: PowerSteering) -> None:
        """Update the statistics display"""
        total_cars = len(ps.cars)
        ffb_cars = sum(1 for car in ps.cars.values() if car.has_custom_ffb())
        undriven_cars = len(ps.undriven_cars)
        predicted_ffb_cars = sum(1 for car in ps.cars.values() if car.ffb_predicted)

        self.stats_text = (
            f"Cars: {total_cars} | "
            f"Custom FFB: {ffb_cars} | "
            f"AI FFB: {predicted_ffb_cars} | "
            f"Undriven: {undriven_cars} | "
            f"Global FFB: {ps.ffb_tarmac}/{ps.ffb_gravel}/{ps.ffb_snow}"
        )
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

class StatsView(BaseCarTableView):
    """View for displaying car statistics"""
    def __init__(self):
        super().__init__()
        self.ps = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(
            DataTable(id="ffb-table")
        )

    def set_powersteering(self, ps: PowerSteering) -> None:
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        if not self.ps:
            self.update("Loading...")
            return

        ffb_table = self._setup_car_table("ffb-table")

        for car in sorted(self.ps.cars.values(), key=lambda x: x.name):
            if car.has_custom_ffb():
                row = self._format_car_row(car)
                ffb_table.add_row(*row)

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
        stats_table.clear(columns=True)
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
        cars_table.clear(columns=True)
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

class UndrivenView(BaseCarTableView):
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

        table = self._setup_car_table("undriven-table")

        for car in sorted(self.ps.undriven_cars.values(), key=lambda x: x.name):
            table.add_row(*self._format_car_row(car))

class PredictionsView(Static):
    """View for displaying FFB predictions"""
    def __init__(self):
        super().__init__()
        self.ps = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="predictions-table")

    def set_powersteering(self, ps: PowerSteering) -> None:
        self.ps = ps
        self._update_display()

    def _update_display(self) -> None:
        if not self.ps:
            self.update("Loading...")
            return

        table = self.query_one("#predictions-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Car", "Weight", "Steering", "Drivetrain", "Current FFB (T/G/S)", "Predicted FFB (T/G/S)", "Status")

        # Train models
        models = self.ps.train_ffb_models()
        if not models:
            table.add_row("No training data available")
            return

        # Get predictions
        cars_with_predictions = self.ps.predict_all_ffb_settings(models)

        for car, predictions in cars_with_predictions:
            current_values = (car.ffb_tarmac, car.ffb_gravel, car.ffb_snow)
            current_ffb = f"{current_values[0]}/{current_values[1]}/{current_values[2]}"
            predicted_ffb = f"{predictions[0]}/{predictions[1]}/{predictions[2]}"

            if car.has_custom_ffb():
                status = "Skipped - Custom FFB"
            else:
                status = "Updated" if current_values != predictions else "Unchanged"

            table.add_row(
                f"{car.id} - {car.name}",
                str(car.weight),
                f"{car.steering_wheel}°",
                car.drive_train,
                current_ffb,
                predicted_ffb,
                status
            )

class MainDisplay(Static):
    """Main display area for car information and operations"""
    def __init__(self):
        super().__init__()
        self.ps = None
        self.stats_view = StatsView()
        self.undriven_view = UndrivenView()
        self.cluster_view = ClusterView()
        self.predictions_view = PredictionsView()
        self.current_view = self.stats_view
        self.last_sort_column = None
        self.last_sort_reverse = False

    def compose(self) -> ComposeResult:
        yield self.stats_view
        yield self.undriven_view
        yield self.cluster_view
        yield self.predictions_view
        self.undriven_view.display = False
        self.cluster_view.display = False
        self.predictions_view.display = False

    def set_powersteering(self, ps: PowerSteering) -> None:
        """Set the PowerSteering instance and update display"""
        self.ps = ps
        self.stats_view.set_powersteering(ps)
        self.undriven_view.set_powersteering(ps)
        self.cluster_view.set_powersteering(ps)
        self.predictions_view.set_powersteering(ps)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort table when header is clicked."""
        def extract_number(value: str) -> float:
            """Extract first number from string, handling kg and degree symbols"""
            import re
            # Remove kg, °, and other non-numeric characters, keeping decimal points
            numbers = re.findall(r'-?\d+\.?\d*', value)
            return float(numbers[0]) if numbers else float('-inf')

        table = event.data_table

        # Toggle sort direction if same column is clicked
        if event.column_key == self.last_sort_column:
            self.last_sort_reverse = not self.last_sort_reverse
        else:
            self.last_sort_reverse = False
            self.last_sort_column = event.column_key

        # Check if this is a numeric column by attempting to extract numbers
        try:
            # If successful, use numeric sorting
            table.sort(event.column_key, key=extract_number, reverse=self.last_sort_reverse)
        except (StopIteration, IndexError, ValueError):
            # Fall back to string sorting for non-numeric columns
            table.sort(event.column_key, reverse=self.last_sort_reverse)

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
        self.predictions_view.display = False
        self.current_view = self.cluster_view

    def show_predictions(self) -> None:
        """Switch to predictions view"""
        self.stats_view.display = False
        self.undriven_view.display = False
        self.cluster_view.display = False
        self.predictions_view.display = True
        self.current_view = self.predictions_view

class PowerSteeringApp(App):
    """A Textual app to manage RSF power steering settings"""

    TITLE = "PowerSteering"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "toggle_stats", "Statistics"),
        ("u", "toggle_undriven", "Undriven Cars"),
        ("c", "toggle_clusters", "Clusters"),
        ("p", "toggle_predictions", "Predictions"),
        ("g", "generate_ffb", "Generate & Apply FFB")
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

    def __init__(self, rsf_path: str, global_ffb: bool = False):
        super().__init__()
        self.rsf_path = rsf_path
        self.global_ffb = global_ffb
        self.ps = PowerSteering(rsf_path, global_ffb)

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

        info_bar.set_stats(self.ps)
        main_display.set_powersteering(self.ps)

    def action_refresh(self) -> None:
        """Reload PowerSteering data and refresh display"""
        self.ps = PowerSteering(self.rsf_path, self.global_ffb)
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

    def action_toggle_predictions(self) -> None:
        """Switch to predictions view"""
        main_display = self.query_one(MainDisplay)
        main_display.show_predictions()
        self.query_one(InfoBar).notify("Generating FFB predictions...")

    async def action_generate_ffb(self) -> None:
        """Generate and apply AI FFB settings with confirmation"""
        info_bar = self.query_one(InfoBar)

        # Train models first
        info_bar.notify("Training FFB models...")
        models = self.ps.train_ffb_models()
        if not models:
            info_bar.notify("Error: No training data available")
            return

        # Generate predictions
        cars_with_predictions = self.ps.predict_all_ffb_settings(models)

        # Generate temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini') as temp_file:
            temp_path = temp_file.name
            self.ps.write_ai_ffb_file(cars_with_predictions, temp_path)

        # Ask for confirmation
        from textual.app import ComposeResult
        from textual.containers import Center
        from textual.screen import ModalScreen
        from textual.widgets import Button, Label

        class ConfirmationModal(ModalScreen[bool]):
            """Screen with a dialog to confirm FFB generation."""

            def compose(self) -> ComposeResult:
                yield Grid(
                    Label("Are you sure you want to apply AI-generated FFB settings?\nThis will backup and replace your personal.ini file.", id="question"),
                    Button("Apply", variant="primary", id="apply"),
                    Button("Cancel", variant="error", id="cancel"),
                    id="dialog",
                )

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "apply":
                    self.dismiss(True)
                else:
                    self.dismiss(False)

        def check_apply(apply: bool | None) -> None:
            """Called when ConfirmationModal is dismissed."""
            if apply:
                try:
                    self.ps.replace_personal_ini_with_predictions(temp_path)
                    info_bar.notify("Successfully applied AI FFB settings")
                    # Refresh to show updated values
                    self.action_refresh()
                except Exception as e:
                    info_bar.notify(f"Error applying FFB settings: {str(e)}")
            else:
                info_bar.notify("FFB generation cancelled")
                import os
                os.unlink(temp_path)

        await self.push_screen(ConfirmationModal(), check_apply)


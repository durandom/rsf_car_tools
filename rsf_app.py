import argparse
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Button, Static
from textual.binding import Binding
from rich.console import Console
from rich.table import Table
import os
from power_poly import Rsf, setup_logging

class RsfDisplay(Static):
    """A custom widget to display RSF statistics and results"""
    pass

class RsfApp(App):
    """RSF Force Feedback Management Application"""

    CSS = """
    Screen {
        align: center middle;
    }

    #controls {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        padding: 1;
        width: 100%;
        height: auto;
    }

    Button {
        width: 100%;
        margin: 1;
    }

    #display {
        height: auto;
        width: 100%;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self, rsf_path: str):
        super().__init__()
        setup_logging(0)  # Set default logging level for the app
        self.rsf = Rsf(rsf_path, record_html=True, gui_mode=True)
        self.models = None

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        yield Container(
            Button("Show Statistics", id="stats", variant="primary"),
            Button("Create Clusters", id="clusters", variant="primary"),
            Button("Train Model", id="train", variant="warning"),
            Button("Generate FFB", id="generate", variant="success"),
            id="controls"
        )
        yield RsfDisplay("", id="display")
        yield Footer()

        # Show initial statistics
        self.call_after_refresh(self.show_statistics)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "stats":
            self.show_statistics()
        elif button_id == "clusters":
            self.create_clusters()
        elif button_id == "train":
            self.train_model()
        elif button_id == "generate":
            self.generate_ffb()

    def show_statistics(self) -> None:
        """Display car statistics"""
        # Capture the console output
        console = Console(record=True)
        with console.capture() as capture:
            self.rsf._log_cars_statistics()

        # Update the display widget
        display = self.query_one("#display")
        display.update_content(console.export_text())

    def create_clusters(self) -> None:
        """Create and display clusters"""
        selected = self.rsf.select_training_sample(3)  # Select 3 cars per cluster

        # Capture the console output
        console = Console(record=True)
        with console.capture() as capture:
            self.rsf.display_selected_sample(selected)

        # Update the display widget
        display = self.query_one("#display")
        display.update_content(console.export_text())

    def train_model(self) -> None:
        """Train the FFB model and display results"""
        self.models = self.rsf.train_ffb_models()

        # Capture the console output
        console = Console(record=True)
        with console.capture() as capture:
            if self.models:
                self.rsf.validate_predictions(self.models)
            else:
                console.print("[red]Failed to train models![/red]")

        # Update the display widget
        display = self.query_one("#display")
        display.update_content(console.export_text())

    def generate_ffb(self) -> None:
        """Generate FFB settings file"""
        if not self.models:
            display = self.query_one("#display")
            display.update_content("[red]Please train the model first![/red]")
            return

        output_file = os.path.join(self.rsf.rsf_path, 'rallysimfans_personal_ai.ini')

        # Capture the console output
        console = Console(record=True)
        with console.capture() as capture:
            self.rsf.generate_ai_ffb_file(self.models, output_file)

        # Update the display widget
        display = self.query_one("#display")
        display.update_content(console.export_text())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RSF Force Feedback Management Application')
    parser.add_argument('--rsf-path', default=os.getcwd(),
                       help='Path to RSF installation directory (default: current working directory)')
    args = parser.parse_args()

    app = RsfApp(args.rsf_path)
    app.run()

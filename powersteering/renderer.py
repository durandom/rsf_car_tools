from rich.console import Console
from rich.table import Table
from typing import List, Tuple, Dict
from loguru import logger
from .models import Car

class ConsoleRenderer:
    """Handles all console output rendering using Rich"""

    def __init__(self, record_html: bool = False, quiet: bool = False):
        """Initialize console renderer

        Args:
            record_html: Whether to record output for HTML export
            quiet: Whether to suppress direct console output
        """
        self.console = Console(record=record_html, quiet=quiet)

    def save_html(self, filename: str) -> None:
        """Save console output to HTML file"""
        self.console.save_html(filename)

    def display_car_statistics(self, cars: Dict[str, Car], undriven_cars: Dict[str, Car],
                             has_custom_ffb_func) -> None:
        """Display statistics about loaded cars"""
        # Create statistics table
        table = Table(title="Car Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        total_cars = len(cars)
        ffb_cars = sum(1 for car in cars.values() if has_custom_ffb_func(car))
        undriven_cars_count = len(undriven_cars)

        table.add_row("Total Cars", str(total_cars))
        table.add_row("Cars with Custom FFB", str(ffb_cars))
        table.add_row("Undriven Cars", str(undriven_cars_count))

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

        # Display undriven cars table
        undriven_table = Table(title="Undriven Cars", show_header=True)
        undriven_table.add_column("Car", style="cyan")
        undriven_table.add_column("Weight", justify="right")
        undriven_table.add_column("Steering", justify="right")
        undriven_table.add_column("Drivetrain")

        for car in sorted(undriven_cars.values(), key=lambda x: x.name):
            undriven_table.add_row(
                f"{car.id} - {car.name}",
                str(car.weight),
                f"{car.steering_wheel}°",
                car.drive_train
            )

        self.console.print(undriven_table)
        self.console.print("\n")

        # Display cars with custom FFB settings
        custom_ffb_table = Table(title="Cars with Custom FFB Settings", show_header=True)
        custom_ffb_table.add_column("Car", style="cyan")
        custom_ffb_table.add_column("Weight", justify="right")
        custom_ffb_table.add_column("Steering", justify="right")
        custom_ffb_table.add_column("Drivetrain")
        custom_ffb_table.add_column("FFB Settings (T/G/S)", justify="right")
        custom_ffb_table.add_column("Global FFB", justify="right")

        for car in sorted(cars.values(), key=lambda x: x.name):
            if has_custom_ffb_func(car):
                custom_ffb_table.add_row(
                    f"{car.id} - {car.name}",
                    str(car.weight),
                    f"{car.steering_wheel}°",
                    car.drive_train,
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                    f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
                )

        self.console.print(custom_ffb_table)
        self.console.print("\n")

    def display_undriven_cars(self, undriven_cars: Dict[str, Car], format_car_details_func) -> None:
        """Display list of undriven cars with key details"""
        table = Table(title="Undriven Cars", show_header=True)
        table.add_column("Car Name", style="cyan")
        table.add_column("Details", style="green")

        for car_id, car in sorted(undriven_cars.items(), key=lambda x: x[1].name):
            table.add_row(
                car.name,
                format_car_details_func(car)
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def display_selected_sample(self, selected_cars: List[Car], cluster_data: Dict) -> None:
        """Display details of selected car sample and cluster statistics"""
        # First display cluster statistics
        cluster_stats = Table(title="Cluster Statistics", show_header=True)
        cluster_stats.add_column("Cluster", justify="right", style="magenta")
        cluster_stats.add_column("Size", justify="right")
        cluster_stats.add_column("Avg Weight", justify="right")
        cluster_stats.add_column("Avg Steering", justify="right")
        cluster_stats.add_column("Drivetrain Distribution")

        from collections import defaultdict

        # Calculate statistics per cluster
        for cluster_id, cars in sorted(cluster_data.items()):
            weights = [car.weight for car in cars]
            min_weight = min(weights)
            max_weight = max(weights)
            weight_dist = f"{min_weight}-{max_weight} kg"

            steering = [car.steering_wheel for car in cars]
            min_steering = min(steering)
            max_steering = max(steering)
            steering_dist = f"{min_steering}-{max_steering}°"

            drive_counts = defaultdict(int)
            for car in cars:
                drive_counts[car.drive_train] += 1
            drive_dist = ", ".join(f"{dt}: {count}" for dt, count in drive_counts.items())

            cluster_stats.add_row(
                str(cluster_id),
                str(len(cars)),
                weight_dist,
                steering_dist,
                drive_dist
            )

        # Add summary statistics
        total_cars = sum(len(cars) for cars in cluster_data.values())
        all_weights = [car.weight for cars in cluster_data.values() for car in cars]
        all_steering = [car.steering_wheel for cars in cluster_data.values() for car in cars]

        weight_dist = f"{min(all_weights)}-{max(all_weights)} kg"
        steering_dist = f"{min(all_steering)}-{max(all_steering)}°"

        all_drive_counts = defaultdict(int)
        for cars in cluster_data.values():
            for car in cars:
                all_drive_counts[car.drive_train] += 1
        all_drive_dist = ", ".join(f"{dt}: {count}" for dt, count in sorted(all_drive_counts.items()))

        cluster_stats.add_row(
            "Total",
            str(total_cars),
            weight_dist,
            steering_dist,
            all_drive_dist,
            style="bold cyan"
        )

        self.console.print("\n")
        self.console.print(cluster_stats)
        self.console.print("\n")

        # Display selected cars table
        table = Table(title=f"Selected Training Sample ({len(cluster_data)} clusters)", show_header=True)
        table.add_column("Cluster", justify="right", style="magenta")
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("FFB (T/G/S)", justify="right")

        current_cluster = None
        for car in selected_cars:
            if current_cluster != car.cluster:
                current_cluster = car.cluster
                
            row_style = "on grey30" if isinstance(current_cluster, int) and current_cluster % 2 == 0 else None
            
            table.add_row(
                str(car.cluster),
                f"{car.id} - {car.name}",
                f"{car.weight}",
                f"{car.steering_wheel}°",
                car.drive_train,
                f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}",
                style=row_style
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def display_ffb_generation_results(self, cars_with_predictions: List[Tuple[Car, Tuple[int, int, int]]],
                                     has_custom_ffb_func) -> None:
        """Display table of FFB generation results"""
        table = Table(title="FFB Generation Results", show_header=True)
        table.add_column("Car", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Steering", justify="right")
        table.add_column("Drivetrain")
        table.add_column("Current FFB (T/G/S)", justify="right")
        table.add_column("Predicted FFB (T/G/S)", justify="right")
        table.add_column("Status")

        for car, predictions in cars_with_predictions:
            row_style = "on red" if has_custom_ffb_func(car) else None

            current_ffb = f"{car.ffb_tarmac}/{car.ffb_gravel}/{car.ffb_snow}"
            predicted_ffb = f"{predictions[0]}/{predictions[1]}/{predictions[2]}"
            status = "Skipped - Custom FFB" if has_custom_ffb_func(car) else "Updated"

            table.add_row(
                f"{car.id} - {car.name}",
                f"{car.weight}",
                f"{car.steering_wheel}°",
                car.drive_train,
                current_ffb,
                predicted_ffb,
                status,
                style=row_style
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
